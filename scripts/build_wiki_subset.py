#!/usr/bin/env python3
"""
Build a curated FAISS index from a subset of Wikipedia articles.

Selects articles relevant to the user's knowledge graph domains, then
builds a small FAISS index that fits in 16 GB RAM.

Usage:
    # Stream from tar.zst (works before full build finishes):
    python scripts/build_wiki_subset.py --from-tar /run/media/lukeh/T9/wiki_embeddings.tar.zst

    # Filter from full build output (faster, needs full build done):
    python scripts/build_wiki_subset.py --from-build

    # Dry run — just count matches, don't extract:
    python scripts/build_wiki_subset.py --from-tar ... --dry-run

    # Limit total articles:
    python scripts/build_wiki_subset.py --from-tar ... --max-articles 50000

Output: wiki_data_subset/ on T9 (or override with --out-dir)
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────────────────
_DATA_ROOT = os.environ.get("WIKI_DATA_ROOT", "/run/media/lukeh/T9")
_DEFAULT_OUT = os.path.join(_DATA_ROOT, "wiki_data_subset")
_DEFAULT_TAR = os.path.join(_DATA_ROOT, "wiki_embeddings.tar.zst")
_FULL_BUILD  = os.path.join(_DATA_ROOT, "wiki_data")

DIM = 384
FLUSH_EVERY = 2_000
ADD_BATCH = 500_000

META_COLS = [
    "id", "title", "text", "chunk_index", "total_chunks",
    "prev_snippet", "next_snippet", "char_start", "char_end",
    "section", "section_level",
]

# ── Domain map ─────────────────────────────────────────────────────
# Derived from user's knowledge graph (262 nodes, 219 edges).
# Each domain has:
#   seeds:    exact article titles to always include
#   keywords: substring matches against article titles (case-insensitive)
#
# Edit freely — this is the main tuning knob for subset size/relevance.

DOMAIN_MAP: Dict[str, Dict[str, List[str]]] = {

    "health_medical": {
        "seeds": [
            "Amantadine", "Lisdexamfetamine", "Diazepam", "Kratom",
            "Honokiol", "Oroxylin A", "Sabroxy", "Benzodiazepine",
            "Bipolar disorder", "Insomnia", "Hypertension",
            "Long COVID", "Schizophrenia", "ADHD",
            "Major depressive disorder", "Generalized anxiety disorder",
            "Substance use disorder", "Melatonin", "Modafinil",
            "Gabapentin", "Clonazepam", "Bupropion", "Sertraline",
            "Cognitive behavioral therapy", "Circadian rhythm",
            "Sleep hygiene", "Norepinephrine", "Acetylcholine",
            "Neuroinflammation", "Blood-brain barrier", "Gut-brain axis",
            "Hypothalamic-pituitary-adrenal axis", "Cortisol",
            "Chronic pain", "Fibromyalgia", "Ehlers-Danlos syndrome",
            "Mast cell", "Histamine", "Vagus nerve",
            "Psychopharmacology", "Drug metabolism", "Cytochrome P450",
            "Receptor (biochemistry)", "Ion channel", "Synapse",
        ],
        "keywords": [
            "pharmacology", "psychiatric", "psychiatry", "antidepressant",
            "anxiolytic", "nootropic", "neurotransmitter", "dopamine",
            "serotonin", "sleep disorder", "neuroplasticity",
            "cardiovascular", "autoimmune", "psychotherapy",
            "herbal medicine", "psychopharmac", "neuroscience",
            "neuropathology", "drug interaction", "clinical pharmacology",
            "mental disorder", "mood disorder", "anxiety disorder",
            "sleep medicine", "pain management", "substance abuse",
        ],
    },

    "education_statistics": {
        "seeds": [
            "Georgia Institute of Technology", "Statistics",
            "Principal component analysis", "Regression analysis",
            "Machine learning", "Data science", "Generalized linear model",
            "Bayesian statistics", "Cross-validation",
            "University of Wisconsin-Madison",
            "Analysis of variance", "Chi-squared test",
            "Student's t-test", "P-value", "Confidence interval",
            "Maximum likelihood estimation", "Expectation-maximization",
            "K-means clustering", "Decision tree learning",
            "Support-vector machine", "Naive Bayes classifier",
            "Ensemble learning", "Boosting (machine learning)",
            "Feature selection", "Curse of dimensionality",
            # Core stats concepts
            "Normal distribution", "Poisson distribution",
            "Binomial distribution", "Exponential distribution",
            "Standard deviation", "Variance", "Correlation",
            "Covariance", "Pearson correlation coefficient",
            "Spearman's rank correlation coefficient",
            "Linear discriminant analysis", "Factor analysis",
            "Multivariate statistics", "Time series",
            "Autoregressive model", "Moving average",
            "Sampling (statistics)", "Central limit theorem",
            "Law of large numbers", "Bootstrap (statistics)",
            "Jackknife resampling", "Permutation test",
            "Kaplan-Meier estimator", "Survival analysis",
            # ML / AI concepts
            "Random forest", "XGBoost", "Gradient boosting",
            "Convolutional neural network", "Recurrent neural network",
            "Long short-term memory", "Autoencoder",
            "Generative adversarial network", "Variational autoencoder",
            "Word2vec", "Word embedding", "Transfer learning",
            "Batch normalization", "Dropout (neural networks)",
            "Stochastic gradient descent", "Adam (optimizer)",
            "Backpropagation", "Perceptron", "Multilayer perceptron",
            "Activation function", "Softmax function",
            "Loss function", "Cross-entropy",
            "Precision and recall", "F-score", "ROC curve",
            "Bias-variance tradeoff", "Regularization (mathematics)",
            "Lasso (statistics)", "Ridge regression",
            "Elastic net regularization",
        ],
        "keywords": [
            "machine learning", "deep learning", "neural network",
            "data mining", "data science",
            "supervised learning", "unsupervised learning",
            "reinforcement learning", "natural language processing",
            "recommender system", "logistic regression",
            "statistical model", "statistical inference",
            "probability distribution", "regression analysis",
            "hypothesis testing", "Bayesian",
        ],
    },

    "computer_science": {
        "seeds": [
            "Python (programming language)", "Linux", "PostgreSQL",
            "SQLite", "Vector database", "Retrieval-augmented generation",
            "Embedding", "FAISS", "Git", "Docker (software)",
            "Transformer (deep learning architecture)",
            "Large language model", "GPT-4", "BERT (language model)",
            "Attention (machine learning)", "Prompt engineering",
            "Semantic search", "Cosine similarity",
            "Locality-sensitive hashing", "Approximate nearest neighbor",
            "ChromaDB", "Redis", "Apache Kafka",
        ],
        "keywords": [
            "programming language", "software engineering",
            "computer science", "data structure",
            "version control", "search engine",
            "information retrieval", "compiler", "operating system",
            "distributed system", "parallel computing",
            "computer network", "graph theory",
            "computational complexity", "artificial intelligence",
            "knowledge graph", "semantic web",
        ],
    },

    "philosophy_mind": {
        "seeds": [
            "Computationalism", "Hedonism", "Utilitarianism",
            "Philosophy of mind", "Functionalism (philosophy of mind)",
            "Atheism", "Existentialism", "Phenomenology (philosophy)",
            "Free will", "Determinism", "Materialism",
            "Epiphenomenalism", "Eliminative materialism",
            "Chinese room", "Turing test", "Hard problem of consciousness",
            "Integrated information theory", "Global workspace theory",
            "Antinatalism", "Negative utilitarianism",
            "David Chalmers", "Daniel Dennett", "Thomas Metzinger",
            "Panpsychism", "Compatibilism", "Moral realism",
            "Effective altruism", "Transhumanism", "Sentience",
        ],
        "keywords": [
            "philosophy of", "ethics", "consciousness",
            "metaphysics", "epistemology",
            "phenomenolog", "existential", "nihilism", "pragmatism",
            "rationalism", "empiricism",
            "qualia", "intentionality", "reductionism",
            "philosophy of science", "philosophy of language",
            "moral philosophy", "political philosophy",
        ],
    },

    "history_politics": {
        "seeds": [
            "Abraham Lincoln", "John Wilkes Booth",
            "Gettysburg Address", "American Civil War",
            "Lithuania", "Bulgaria", "History of Chicago",
            "Reconstruction era", "Emancipation Proclamation",
            "Grand Duchy of Lithuania", "Bulgarian Empire",
            "History of Lithuania", "History of Bulgaria",
            "Immigration to the United States", "Lithuanian Americans",
            "Polish-Lithuanian Commonwealth",
        ],
        "keywords": [
            "Lithuanian", "Baltic state", "Bulgarian",
            "political philosophy", "authoritarianism",
        ],
    },

    "geography_places": {
        "seeds": [
            "Chicago", "Naperville, Illinois", "Indiana",
            "Kansas City", "Atlanta", "Logan Square, Chicago",
            "Spain", "Lithuania", "Bulgaria", "Colorado",
            "DuPage County, Illinois", "Cook County, Illinois",
        ],
        # No keywords — seeds only. "Illinois"/"Chicago" are too broad.
        "keywords": [],
    },

    "hobbies_interests": {
        "seeds": [
            "Hearts of Iron IV", "Brewing", "Beer", "Trappist beer",
            "Rochefort Brewery", "Paczki", "Pumpernickel",
            "Idles (band)", "Punk rock", "Post-punk",
            "Europa Universalis IV", "Crusader Kings III",
            "Paradox Interactive", "World War II",
            "Homebrewing", "Ale", "Lager", "Belgian beer",
            "Fermentation in food processing",
            "Sourdough", "Rye bread",
            "Fugazi", "Black Flag (band)", "Minor Threat",
            "Hardcore punk", "Noise rock", "Shoegaze",
        ],
        "keywords": [
            "strategy game", "grand strategy",
            "craft beer", "beer style", "homebrewing",
            "punk rock", "post-punk", "indie rock",
            "songwriting", "music theory",
        ],
    },

    "psychology_relationships": {
        "seeds": [
            "Attachment theory", "Family systems theory",
            "Codependency", "Emotional intelligence",
            "Cognitive behavioral therapy", "Dialectical behavior therapy",
            "Schema therapy", "Internal Family Systems",
            "Borderline personality disorder", "Complex PTSD",
            "Adverse childhood experiences", "Polyvagal theory",
            "Learned helplessness", "Self-determination theory",
            "Flow (psychology)", "Maslow's hierarchy of needs",
            "Erikson's stages of psychosocial development",
            "Object relations theory", "Locus of control",
            "Dunning-Kruger effect", "Cognitive dissonance",
            "Confirmation bias", "Availability heuristic",
            "Prospect theory", "Hedonic treadmill",
        ],
        "keywords": [
            "psychology", "cognitive bias", "behavioral economics",
            "personality disorder", "psychotherapy",
            "developmental psychology", "social psychology",
            "clinical psychology", "neuropsychology",
            "cognitive science", "decision making",
        ],
    },

    # ── Broad cross-domain terms (for synthesis pipeline diversity) ──
    # These match the _WIKI_QUERY_SEEDS in synthesis_generator.py

    "cross_domain_science": {
        "seeds": [
            # ── Systems / complexity ──
            "Systems theory", "Feedback", "Emergence",
            "Complex system", "Network theory", "Information theory",
            "Entropy", "Game theory", "Chaos theory",
            "Autopoiesis", "Dissipative system", "Attractor",
            "Fitness landscape", "Red Queen hypothesis",
            "Catastrophe theory", "Tipping point (sociology)",
            "Path dependence", "Hysteresis", "Resilience (ecology)",
            # ── Math / probability ──
            "Central limit theorem", "Bayes' theorem",
            "Markov chain", "Monte Carlo method",
            "Fourier transform", "Signal-to-noise ratio",
            "Normal distribution", "Power law", "Zipf's law",
            "Benford's law", "Law of large numbers",
            "Random walk", "Brownian motion", "Percolation theory",
            "Graph theory", "Small-world network", "Scale-free network",
            # ── Biology / ecology ──
            "Evolution", "Natural selection", "Adaptation",
            "Homeostasis", "Allostasis", "Hormesis",
            "Mutualism (biology)", "Symbiosis", "Parasitism",
            "Commensalism", "Trophic level", "Food web",
            "Keystone species", "Island biogeography", "Ecological niche",
            "Metabolic pathway", "Enzyme kinetics",
            "Action potential", "Hebbian theory",
            "Epigenetics", "Gene expression", "Phenotypic plasticity",
            "Circadian rhythm", "Biological clock",
            "Immune system", "Antibody", "Microbiome",
            "Apoptosis", "Mitosis", "Stem cell",
            "Photosynthesis", "Cellular respiration",
            "Predator-prey", "Lotka-Volterra equations",
            # ── Optimization / algorithms ──
            "Ant colony optimization", "Particle swarm optimization",
            "Simulated annealing", "Genetic algorithm",
            "Stigmergy", "Quorum sensing",
            "Gradient descent", "Convex optimization",
            "Constraint satisfaction", "Traveling salesman problem",
            # ── Economics / game theory ──
            "Supply and demand", "Diminishing returns",
            "Tragedy of the commons", "Public goods game",
            "Prisoner's dilemma", "Nash equilibrium",
            "Pareto efficiency", "Arrow's impossibility theorem",
            "Comparative advantage", "Opportunity cost",
            "Externality", "Market failure", "Moral hazard",
            "Principal-agent problem", "Adverse selection",
            # ── Physics / thermodynamics ──
            "Thermodynamics", "Second law of thermodynamics",
            "Maxwell's demon", "Entropy (information theory)",
            "Phase transition", "Critical point (thermodynamics)",
            "Resonance", "Harmonic oscillator", "Wave",
            "Diffusion", "Osmosis", "Viscosity",
            # ── Broad field articles (guaranteed inclusion) ──
            "Biology", "Ecology", "Physics", "Chemistry",
            "Mathematics", "Economics", "Sociology", "Anthropology",
            "Linguistics", "Engineering", "Neuroscience",
            "Genetics", "Immunology", "Microbiology", "Botany",
            "Zoology", "Geology", "Astronomy", "Climatology",
        ],
        "keywords": [
            "biomimicry", "biomimetic", "cybernetics",
            "control theory", "signal processing",
            "network science", "scale-free", "small-world",
            "dynamical system", "nonlinear dynamics",
            "genetic algorithm", "swarm intelligence",
            "cellular automaton", "agent-based model",
            "phase transition", "self-organization",
            "power law", "complex adaptive",
            "evolutionary biology", "population genetics",
            "systems biology", "computational biology",
            "ecological model", "mathematical biology",
            "information entropy", "thermodynamic entropy",
        ],
    },

    # ── Verification pair articles (must be present for testing) ──

    "verification_pairs": {
        "seeds": [
            "Velcro", "Burdock", "PageRank", "Citation analysis",
            "Simulated annealing", "Annealing (metallurgy)",
            "Kingfisher", "Shinkansen", "Ant colony optimization",
            "Foraging", "Information theory",
            "Entropy (thermodynamics)", "Lotka-Volterra equations",
            "Predation", "Nervous system", "Internet",
            "Termite mound", "Air conditioning",
            "Stigmergy", "Version control", "Quorum sensing",
            "Consensus algorithm",
            # FALSE pairs (need these to test coherence rejection)
            "Jazz", "Plate tectonics", "Photosynthesis",
            "Stock market", "Knitting", "General relativity",
            "Great Wall of China", "DNA replication",
            "Quantum mechanics", "Sourdough",
        ],
        "keywords": [],
    },
}


# ── Matching engine ────────────────────────────────────────────────

def compile_matcher(domain_map: Dict) -> Tuple[Set[str], List[re.Pattern]]:
    """Compile domain map into fast matching structures.

    Returns:
        seeds: set of exact titles (case-normalized)
        patterns: list of compiled regex patterns for keyword matching
    """
    seeds: Set[str] = set()
    keyword_set: Set[str] = set()

    for domain, spec in domain_map.items():
        for s in spec.get("seeds", []):
            seeds.add(s.lower().strip())
        for k in spec.get("keywords", []):
            keyword_set.add(k.lower().strip())

    # Build regex patterns for keywords — match as substring, word-boundary-ish
    # For short keywords (<=4 chars), require word boundaries to avoid false matches
    patterns = []
    for kw in sorted(keyword_set):
        if len(kw) <= 4:
            patterns.append(re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE))
        else:
            patterns.append(re.compile(re.escape(kw), re.IGNORECASE))

    return seeds, patterns


def title_matches(title: str, seeds: Set[str], patterns: List[re.Pattern]) -> bool:
    """Check if a Wikipedia article title matches any domain term."""
    t_lower = title.lower().strip()

    # Exact seed match
    if t_lower in seeds:
        return True

    # Keyword substring match
    for pat in patterns:
        if pat.search(title):
            return True

    return False


# ── Streaming from tar.zst ─────────────────────────────────────────

def stream_tar(archive_path: str):
    """Yield (filename, DataFrame) from tar.zst."""
    proc = subprocess.Popen(
        ["zstd", "-d", "-c", archive_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=4 * 1024 * 1024,
    )
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".parquet"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    yield pd.read_parquet(io.BytesIO(f.read()))
                except Exception:
                    pass
    finally:
        proc.terminate()
        proc.wait()


# ── Build subset ───────────────────────────────────────────────────

def build_subset_from_tar(tar_path: str, out_dir: str,
                          seeds: Set[str], patterns: List[re.Pattern],
                          max_articles: int = 0, dry_run: bool = False):
    """Stream tar.zst, select matching articles, build FAISS subset."""
    os.makedirs(out_dir, exist_ok=True)

    embed_file = os.path.join(out_dir, "embeddings_mmap.dat")
    meta_file  = os.path.join(out_dir, "metadata.parquet")
    faiss_file = os.path.join(out_dir, "vector_index_ivf.faiss")
    meta_dir   = os.path.join(out_dir, "_meta_parts")
    index_meta = os.path.join(out_dir, "index_meta.json")

    matched_titles: Set[str] = set()
    skipped_titles: Set[str] = set()
    file_count = 0
    match_count = 0
    idx_offset = 0
    part_num = 0
    meta_batch = []
    domain_counts: Dict[str, int] = {}

    emb_f = None if dry_run else open(embed_file, "wb")
    t0 = time.time()

    print(f"Streaming {tar_path} ...")
    print(f"Seeds: {len(seeds)}, Keyword patterns: {len(patterns)}")
    if max_articles:
        print(f"Max articles: {max_articles:,}")
    print()

    for df in stream_tar(tar_path):
        file_count += 1
        title = df["title"].iloc[0] if "title" in df.columns else ""

        if not title or title in skipped_titles:
            if file_count % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  scanned {file_count:,} | matched {match_count:,} "
                      f"({len(matched_titles):,} articles) | {elapsed/60:.1f}min")
            continue

        if title in matched_titles:
            # Already matched this article — include all its chunks
            pass
        elif title_matches(title, seeds, patterns):
            matched_titles.add(title)
            match_count += 1
        else:
            skipped_titles.add(title)
            continue

        if max_articles and len(matched_titles) > max_articles:
            if title not in matched_titles:
                skipped_titles.add(title)
                continue

        if dry_run:
            continue

        # ── Write embeddings ──
        n = len(df)
        emb = np.vstack(df["embedding"].values).astype("float32")
        emb_f.write(emb.tobytes())

        # ── Collect metadata ──
        cols = [c for c in META_COLS if c in df.columns]
        bdf = df[cols].copy()
        bdf.insert(0, "idx", range(idx_offset, idx_offset + n))
        bdf["source"] = "wikipedia"
        meta_batch.append(bdf)
        idx_offset += n

        # ── Periodic flush ──
        if match_count % FLUSH_EVERY == 0 and meta_batch:
            os.makedirs(meta_dir, exist_ok=True)
            combined = pd.concat(meta_batch, ignore_index=True)
            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(table, os.path.join(meta_dir, f"part_{part_num:05d}.parquet"),
                           compression="zstd")
            meta_batch.clear()
            part_num += 1
            emb_f.flush()

            elapsed = time.time() - t0
            print(f"  scanned {file_count:,} | matched {match_count:,} "
                  f"({len(matched_titles):,} articles, {idx_offset:,} vectors, "
                  f"{idx_offset * DIM * 4 / 1e9:.1f} GB) | {elapsed/60:.1f}min")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\nScan complete: {file_count:,} files in {elapsed/60:.1f}min")
    print(f"  Matched articles:  {len(matched_titles):,}")
    print(f"  Matched chunks:    {idx_offset:,}")
    print(f"  Skipped articles:  {len(skipped_titles):,}")

    if dry_run:
        print("\n(Dry run — no files written)")
        # Print sample matched titles
        sample = sorted(matched_titles)[:50]
        print(f"\nSample matched titles ({len(sample)} of {len(matched_titles)}):")
        for t in sample:
            print(f"  {t}")
        return 0

    # ── Final metadata flush ──
    if meta_batch:
        os.makedirs(meta_dir, exist_ok=True)
        combined = pd.concat(meta_batch, ignore_index=True)
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, os.path.join(meta_dir, f"part_{part_num:05d}.parquet"),
                       compression="zstd")
    emb_f.close()

    # ── Merge metadata parts ──
    parts = sorted(Path(meta_dir).glob("part_*.parquet"))
    if parts:
        print(f"\nMerging {len(parts)} metadata parts ...")
        schema = pq.read_schema(parts[0])
        writer = pq.ParquetWriter(meta_file, schema, compression="zstd")
        for p in parts:
            writer.write_table(pq.read_table(p))
        writer.close()
        for p in parts:
            p.unlink()
        Path(meta_dir).rmdir()

    # ── Save index meta ──
    with open(index_meta, "w") as f:
        json.dump({
            "total_vectors": idx_offset,
            "embedding_dim": DIM,
            "articles_matched": len(matched_titles),
            "files_scanned": file_count,
            "matched_titles": sorted(matched_titles),
        }, f, indent=2)

    print(f"\n  Embeddings: {embed_file} ({os.path.getsize(embed_file)/1e9:.1f} GB)")
    print(f"  Metadata:   {meta_file}")

    # ── Build FAISS ──
    build_faiss(out_dir, idx_offset)

    return idx_offset


# ── FAISS builder ──────────────────────────────────────────────────

def build_faiss(out_dir: str, total_vectors: int = 0):
    """Build FAISS IVF index from subset embeddings."""
    embed_file = os.path.join(out_dir, "embeddings_mmap.dat")
    faiss_file = os.path.join(out_dir, "vector_index_ivf.faiss")
    index_meta = os.path.join(out_dir, "index_meta.json")

    if total_vectors == 0:
        if os.path.exists(index_meta):
            with open(index_meta) as f:
                total_vectors = json.load(f)["total_vectors"]
        else:
            total_vectors = os.path.getsize(embed_file) // (DIM * 4)

    print(f"\nBuilding FAISS IVF for {total_vectors:,} vectors ...")

    emb = np.memmap(embed_file, dtype="float32", mode="r",
                    shape=(total_vectors, DIM))

    # Scale centroids to dataset size
    nlist = min(int(4 * np.sqrt(total_vectors)), total_vectors // 39)
    nlist = max(nlist, 1)
    train_size = min(total_vectors, max(nlist * 39, 50_000))

    print(f"  Centroids: {nlist:,}, training on {train_size:,} samples")

    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist)

    rng = np.random.default_rng(42)
    train_idx = rng.choice(total_vectors, train_size, replace=False)
    train_idx.sort()
    train_data = np.array(emb[train_idx])

    print("  Training ...")
    t0 = time.time()
    index.train(train_data)
    print(f"  Trained in {time.time() - t0:.1f}s")
    del train_data

    print("  Adding vectors ...")
    for start in range(0, total_vectors, ADD_BATCH):
        end = min(start + ADD_BATCH, total_vectors)
        index.add(np.array(emb[start:end]))
        print(f"    {end:,} / {total_vectors:,}")

    index.nprobe = min(32, nlist)
    faiss.write_index(index, faiss_file)
    print(f"  Saved: {faiss_file} ({index.ntotal:,} vectors)")


# ── Filter from full build ─────────────────────────────────────────

def build_subset_from_full(build_dir: str, out_dir: str,
                           seeds: Set[str], patterns: List[re.Pattern],
                           max_articles: int = 0):
    """Filter from completed full build (metadata + binary embeddings)."""
    meta_path = os.path.join(build_dir, "metadata.parquet")
    embed_path = os.path.join(build_dir, "embeddings_mmap.dat")

    if not os.path.exists(meta_path):
        sys.exit(f"Full build metadata not found: {meta_path}")

    os.makedirs(out_dir, exist_ok=True)
    out_embed = os.path.join(out_dir, "embeddings_mmap.dat")
    out_meta  = os.path.join(out_dir, "metadata.parquet")
    index_meta = os.path.join(out_dir, "index_meta.json")

    print("Scanning full build metadata for matching titles ...")
    print(f"  Seeds: {len(seeds)}, Patterns: {len(patterns)}")

    # Read only title + idx columns (memory efficient)
    pf = pq.ParquetFile(meta_path)
    matching_indices = []
    matched_titles: Set[str] = set()
    total_rows = 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["idx", "title"]):
        df = batch.to_pandas()
        total_rows += len(df)

        for _, row in df.iterrows():
            title = row["title"]
            if title in matched_titles or title_matches(title, seeds, patterns):
                matched_titles.add(title)
                matching_indices.append(int(row["idx"]))

                if max_articles and len(matched_titles) > max_articles:
                    break

        if total_rows % 5_000_000 == 0:
            print(f"  Scanned {total_rows:,} rows, matched {len(matched_titles):,} articles")

        if max_articles and len(matched_titles) > max_articles:
            break

    print(f"\n  Total rows scanned: {total_rows:,}")
    print(f"  Matched articles:  {len(matched_titles):,}")
    print(f"  Matched chunks:    {len(matching_indices):,}")

    # Extract matching rows
    print("\nExtracting matching embeddings ...")
    matching_indices.sort()
    idx_set = set(matching_indices)

    # Read embeddings by index from binary file
    full_emb_size = os.path.getsize(embed_path) // (DIM * 4)
    full_emb = np.memmap(embed_path, dtype="float32", mode="r",
                         shape=(full_emb_size, DIM))

    with open(out_embed, "wb") as f:
        for i, idx in enumerate(matching_indices):
            f.write(full_emb[idx].tobytes())
            if (i + 1) % 500_000 == 0:
                print(f"  Extracted {i+1:,} / {len(matching_indices):,}")

    # Extract matching metadata
    print("Extracting matching metadata ...")
    writer = None
    new_idx = 0
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        mask = df["idx"].isin(idx_set)
        if mask.any():
            subset = df[mask].copy()
            subset["idx"] = range(new_idx, new_idx + len(subset))
            new_idx += len(subset)
            table = pa.Table.from_pandas(subset, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_meta, table.schema, compression="zstd")
            writer.write_table(table)
    if writer:
        writer.close()

    # Save index meta
    with open(index_meta, "w") as f:
        json.dump({
            "total_vectors": len(matching_indices),
            "embedding_dim": DIM,
            "articles_matched": len(matched_titles),
            "matched_titles": sorted(matched_titles),
        }, f, indent=2)

    print(f"\n  Embeddings: {out_embed} ({os.path.getsize(out_embed)/1e9:.1f} GB)")
    print(f"  Metadata:   {out_meta}")

    build_faiss(out_dir, len(matching_indices))


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-tar", metavar="PATH",
                      help="Stream from tar.zst archive")
    mode.add_argument("--from-build", action="store_true",
                      help="Filter from completed full build")
    p.add_argument("--out-dir", default=_DEFAULT_OUT,
                   help=f"Output directory (default: {_DEFAULT_OUT})")
    p.add_argument("--max-articles", type=int, default=0,
                   help="Cap total articles (0=unlimited)")
    p.add_argument("--dry-run", action="store_true",
                   help="Count matches only, don't extract")
    p.add_argument("--faiss-only", action="store_true",
                   help="Just build FAISS from existing subset embeddings")
    args = p.parse_args()

    if args.faiss_only:
        build_faiss(args.out_dir)
        sys.exit(0)

    seeds, patterns = compile_matcher(DOMAIN_MAP)

    if args.from_tar:
        build_subset_from_tar(args.from_tar, args.out_dir, seeds, patterns,
                              max_articles=args.max_articles,
                              dry_run=args.dry_run)
    else:
        build_subset_from_full(_FULL_BUILD, args.out_dir, seeds, patterns,
                               max_articles=args.max_articles)

    print("\nDone!")
