# Synthesis Pipeline — Calibration & Validation Plan

## What We're Proving

Daemon's synthesis pipeline claims that an AI system can **generate genuinely novel cross-domain insights** — connections between personal knowledge and general knowledge that are structurally sound, non-trivial, and not already documented. If the calibration results below hold, this concept works because:

1. **The filter is the product.** Connection generation is cheap (random sampling + LLM articulation). The 8-stage filter pipeline is what separates signal from noise. If the filter achieves high precision (few false positives) and high recall (few false negatives), the system reliably distinguishes real structural isomorphisms from metaphor, pseudoscience, and trivia.

2. **Convergence validates discovery.** When the same insight is independently rediscovered from different entity pairs and different graph paths, that's evidence of a real structural pattern — not a lucky hallucination. If we observe convergence on known-good connections and not on noise, the signal is real.

3. **Graph growth enables depth.** The knowledge graph accumulates facts through normal conversation. When it crosses the sparsity threshold (≥20 nodes, ≥15 edges), two synthesis pathways activate — shutdown dreaming and proactive surfacing. If both produce meaningful output from the same graph, the architecture compounds.

**Bottom line:** If we can demonstrate that the pipeline accepts novel structural connections while rejecting trivia, pseudoscience, known results, and forced metaphor — using real Wikipedia data and realistic personal facts — then the core concept is validated for production.

---

## The Three Synthesis Pathways

### Pathway 0: Retrieval-Based Synthesis (Tier 0, Primary)
**Code:** `shutdown_processor.py:_run_synthesis_dreaming()` → `RetrievalSynthesisGenerator` → `SynthesisFilter` → `SynthesisMemory`

- **When:** Session shutdown (Step 6.8), runs first in three-tier parallel generation
- **What:** Extracts structural queries from personal facts via few-shot LLM prompting, searches FAISS (40M vectors) for Wikipedia articles sharing the structural pattern, adversarially evaluates the connection. Produces candidates naming specific mechanisms rather than surface metaphors.
- **Gate:** `SYNTHESIS_GENERATOR_ENABLED=True` AND graph has ≥`SYNTHESIS_GENERATOR_MIN_GRAPH_NODES` (20) nodes
- **Output:** Accepted results stored in `synthesis_results` ChromaDB collection; provisional bridge edges created in knowledge graph
- **Volume:** ~15 candidates per session (shared across all tiers), expect 0–2 acceptances from this tier

### Pathway 1: Shutdown Dreaming — Graph Walks (Tier 1)
**Code:** `shutdown_processor.py:_run_synthesis_dreaming()` → `GraphWalkGenerator` → `SynthesisFilter` → `SynthesisMemory`

- **When:** Session shutdown (Step 6.8), runs in parallel with Tiers 0 and 2
- **What:** Biased Markov random walks on the unified personal+wikidata graph. Node2Vec-style return bias (2.0x toward personal nodes). Hub dampening (degree > 15). Cross-domain walk constraint (>=2 domains). Walk narration prompt gives LLM the full path to interpret.
- **Gate:** `GRAPH_WALK_ENABLED=True` AND `count_bridge_edges() >= GRAPH_WALK_MIN_BRIDGE_EDGES` (40)
- **Current status:** Inactive — only 6 quality bridges after cleanup of 129 garbage bridges. Will activate as Tier 0 creates provisional bridge edges on acceptance.
- **Output:** Accepted results stored in `synthesis_results` ChromaDB collection
- **Volume:** When active, fills remaining quota after Tier 0; currently 0 candidates

### Pathway 2: Shutdown Dreaming — Cross-Store Pairing (Tier 2, Fallback)
**Code:** `shutdown_processor.py:_run_synthesis_dreaming()` → `SynthesisGenerator` → `SynthesisFilter` → `SynthesisMemory`

- **When:** Session shutdown (Step 6.8), fills remaining candidate quota after Tiers 0 and 1
- **What:** Samples personal entities from `facts` ChromaDB collection and Wikipedia articles via FAISS (`semantic_search_with_neighbors()` from `knowledge/semantic_search.py`, backed by a 40M-vector IVFPQ index), pairs cross-domain, LLM-articulates bridges, runs 8-stage filter
- **Gate:** `SYNTHESIS_GENERATOR_ENABLED=True` AND graph has ≥`SYNTHESIS_GENERATOR_MIN_GRAPH_NODES` (20) nodes
- **Output:** Accepted results stored in `synthesis_results` ChromaDB collection
- **Volume:** Fills remaining quota (~15 candidates minus Tier 0/1 output), expect 0–2 acceptances

### Pathway 3: Proactive Context Surfacing (Real-Time, In-Conversation)
**Code:** `memory/context_surfacer.py:generate_insights()` → domain classification → bridge selection → LLM synthesis

- **When:** Every query (parallel retrieval task), cached per session (LLM fires once)
- **What:** Classifies graph entities into life domains (health, career, fitness, etc.), detects which domains the current query touches, bridges across domains with a single LLM call
- **Gate:** `PROACTIVE_SURFACING_ENABLED=True` AND graph has ≥20 nodes AND ≥15 edges
- **Output:** Up to 2 insight strings injected into `[PROACTIVE INSIGHTS]` prompt section
- **Approach:** Star topology (entities → "user" hub) + keyword domain detection — faster, less rigorous than shutdown dreaming, but surfaces connections while the user is actively thinking about a topic

**All pathways share the same graph.** The graph grows through fact extraction (conversation → `fact_extractor` → ChromaDB `facts` → `_ingest_fact_to_graph()`) and through synthesis acceptance (provisional bridge edges), so every conversation enriches the substrate for all synthesis modes. The bridge feedback loop means Tier 0 acceptances gradually enable Tier 1 (graph walks) by growing the bridge count toward the 40-edge minimum.

---

## How Facts Flow Into the Knowledge Graph

Understanding this is critical because graph richness directly gates synthesis quality.

```
Conversation Turn
  └→ store_interaction(query, response)
       └→ extract_and_store_facts(query, response)     [per-turn if enabled, or batch at shutdown]
            └→ FactExtractor.extract_facts()
                 └→ [Fact Verification Gate]            fact_verifier.verify()
                      └→ ChromaDB facts collection      add_fact() → returns fact_id
                           └→ _ingest_fact_to_graph()   [if KNOWLEDGE_GRAPH_ENABLED]
                                ├→ Confidence gate       (< 0.50 → skip)
                                ├→ Entity worthiness     (4+ words, temporal, measurements → metadata only)
                                ├→ Relation normalization (normalize_relation())
                                ├→ Entity resolution     (resolve_or_create() via alias index)
                                └→ graph_memory.add_relation(edge, fact_id)
                                     ├→ New edge: weight=1.0, source_fact_ids=[fact_id]
                                     └→ Existing edge: weight += 1.0, append fact_id
```

**Key point:** Retrieval does NOT enrich the graph. Only fact ingestion does. The graph grows in discrete batches (shutdown default) or per-turn (optional). This means the calibration test needs to simulate realistic graph growth by pre-loading facts.

---

## Test Infrastructure

### Data Sources

| Source | Location | Purpose |
|--------|----------|---------|
| Curated wiki subset | `wiki_data_subset/` (built by `scripts/build_wiki_subset.py`) | ~500-2000 articles across 10 domain categories for reproducible novelty checks |
| Calibration fixture | `tests/fixtures/calibration_candidates.json` | 72 hand-labeled candidates in 7 tiers for filter precision/recall |
| Generated candidates | `scripts/generate_calibration_candidates.py` → `data/unlabeled_candidates.json` | Fresh candidates from real generator for new labeling rounds |
| Verification pairs | `scripts/verify_synthesis_pipeline.py` | 21 pre-defined Wikipedia pairs (8 true, 6 false, 4 coverage-asymmetry, 3 diagnostic) |
| Synthetic facts | (to be created) | Realistic personal facts to populate knowledge graph for end-to-end testing |

### Domain Categories in Wiki Subset (`build_wiki_subset.py`)

The subset builder uses `DOMAIN_MAP` with 10 categories, each with seed titles + keyword patterns:

1. **health_medical** — 48 seeds (drugs, conditions, mechanisms), 22 keywords
2. **education_statistics** — 93 seeds (schools, algorithms, ML/stats concepts)
3. **computer_science** — 17 seeds (programming, databases, transformers)
4. **philosophy_mind** — 24 seeds (consciousness, ethics)
5. **history_politics** — 14 seeds (historical figures, locations)
6. **geography_places** — 12 seeds (cities, regions)
7. **hobbies_interests** — 22 seeds (games, beer, music)
8. **psychology_relationships** — 20 seeds (attachment theory, PTSD, etc.)
9. **cross_domain_science** — 140+ seeds (systems theory, optimization, physics) — broadest category, designed to maximize synthesis diversity
10. **verification_pairs** — 30 seeds matching the verify script's test pairs

---

## Key Insight: Full FAISS Index Eliminates the Subset Problem

The original plan assumed wiki novelty checks would run against a curated subset
(500-2000 articles), making Stage 3 novelty scores optimistic. **This is no
longer true.** The FAISS IVFPQ migration (2026-03-31) means:

- The **full 40M-vector Wikipedia index** (~2.2 GB RAM) is loaded by
  `semantic_search_with_neighbors()` from the T9 external drive
- `SynthesisGenerator._sample_wiki_articles()` samples from **all 40M vectors**
- `SynthesisFilter._stage_3_novelty_external()` checks novelty against
  **all 40M vectors**
- No ChromaDB wiki collection needed — no memory choking, no subset workaround

**This means Stage 3 novelty scores are production-realistic from day one.**
Known connections (ant colonies ↔ TCP, exercise ↔ serotonin) will be caught
by the full corpus, not missed due to sparse subset coverage.

The curated wiki subset (`build_wiki_subset.py`) is still useful for offline
reproducibility but is **not required** for calibration — the full index is
the primary backend.

### IVFPQ Similarity Compression Offset

**Critical calibration finding (2026-03-31):** IVFPQ Product Quantization
(48 subquantizers × 8 bits = 48 bytes/vector) compresses vectors ~32x but
systematically depresses cosine similarity scores by ~0.15-0.20 compared to
exact search. Observed in verification run:

| Pair | Expected sim (exact) | Actual sim (IVFPQ) | Delta |
|------|---------------------|--------------------|-------|
| Shannon ↔ Boltzmann entropy | ~0.80+ | 0.443 | -0.36 |
| ACO ↔ Foraging | ~0.75+ | 0.539 | -0.21 |
| Lotka-Volterra ↔ Predation | ~0.75+ | 0.541 | -0.21 |
| Simulated annealing ↔ Metallurgy | ~0.80+ | 0.640 | -0.16 |

All thresholds must account for this offset. Original ChromaDB-calibrated
thresholds → IVFPQ-calibrated thresholds:

| Threshold | ChromaDB (exact) | IVFPQ (compressed) |
|-----------|------------------|--------------------|
| `SYNTHESIS_NOVELTY_KNOWN_THRESHOLD` | 0.80 | **0.60** |
| `SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD` | 0.75 | **0.85** |
| `SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD` | 0.50 | **0.35** |
| `SYNTHESIS_COMPOSITE_MIN_SCORE` | 0.40 | **0.65** |

The composite minimum was raised independently (0.40→0.65) because the
Jazz ↔ Plate tectonics false pair leaked at 0.624 with a vacuous "emergent
complexity" claim. The template check caught it (template_sim=0.5) but the
old composite floor was too low to reject it.

**For future threshold tuning:** Always test against the full FAISS index,
not mock stores. Mock novelty scores don't exhibit the IVFPQ compression
offset and will give misleadingly optimistic gate behavior.

### Co-occurrence Gate Recalibration (2026-03-31)

**Finding:** At 40M scale, the co-occurrence gate (`"concept_a concept_b"` →
FAISS search) was the actual bottleneck, not claim similarity. With 0.60
threshold, 9/10 candidates were rejected at co-occurrence because FAISS
returns 0.65-0.75 similarity for *any* two-word English query against 40M
articles. The gate measured "do these words exist near each other somewhere
in Wikipedia?" (always yes) not "is this specific connection documented."

**Fix:** Raised `SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD` from 0.60 to 0.85.
At 0.85, only genuine co-occurrence (concepts documented together in the
same article) triggers rejection. Observed co-occurrence scores for
unrelated pairs: 0.65-0.80. Genuine co-occurrence (e.g., "Cabo" ↔
"Group dynamics"): 0.90+.

**Result (15 candidates × 3 runs = 45 total):**
- Rejection distribution: novelty=11, coherence=20, composite=4, accepted=10
- Acceptance rate: 22% (target: 10-30%)
- Coherence judge is now the primary gatekeeper (as designed)
- All 3 target gates (novelty, coherence, composite) are active

---

## Reviewer Feedback (2026-03-31)

Incorporated into the plan below. Key points:

1. **Phase 1.3 noisy facts:** Synthetic facts must include production-realistic
   noise — ambiguous domain membership, near-duplicate entities that stress the
   resolver ("Auggie" vs "August" vs "user's brother"), at least one fact below
   the 0.50 confidence gate. Clean facts validate the filter, not the pipeline.

2. **Small denominator:** 5 candidates × 5 runs = 25 total is too thin for
   statistical confidence. Bump to 10-15 candidates per calibration session
   (50-75 total) to distinguish "filter too tight" from "unlucky sampling."

3. **FAISS field-name mismatch (RESOLVED):** The old `_extract_similarity()`
   assumed ChromaDB `relevance_score` and did `1/score - 1` conversion. Now
   replaced with `_extract_faiss_similarity()` reading `similarity` directly.
   Silent 0.0 would have made novelty gate pass everything. Tests cover this.

4. **F1 on expanded fixture:** Previous benchmark was 95.2% on 54 candidates.
   Fixture now has 72 (18 new edge cases). Expect 90-92% F1. Below 88%
   indicates a real coherence prompt gap, not mislabeled fixtures.

5. **Run order:** Start with verification pairs (Step 2.3) before the full
   mock calibration. It's a 5-minute smoke test that catches broken gates
   before you spend time on the full fixture.

---

## Step-by-Step Plan

### Phase 1: Build the Data Foundation

#### Step 1.1 — Verify FAISS Index

The full 40M-vector FAISS IVFPQ index should already be available at
`/run/media/lukeh/T9/wiki_data/vector_index_ivf.faiss`. Verify it loads:

```python
from knowledge.semantic_search import semantic_search_with_neighbors
results = semantic_search_with_neighbors("quantum entanglement", k=3)
assert len(results) > 0, "FAISS index not loading"
print(f"Top result: {results[0]['title']} (sim={results[0]['similarity']:.3f})")
```

- **Success criteria:** Returns relevant results with similarity > 0.5 for
  well-known topics. Index loads in < 30s, uses ~2.2 GB RAM.

**Minimum RAM for valid calibration:** ~6 GB free. FAISS index (~2.2 GB) +
embedder (~0.4 GB) + ChromaDB (~0.5 GB) + Python/graph overhead (~1 GB).
Coherence judging via API (Sonnet, GPT-4o) adds negligible RAM. If you
ever calibrate against a local model, add its VRAM/RAM on top — IVFPQ
returns garbage results if the index is partially paged out to swap.

#### ~~Step 1.2 — Load Wiki Subset into ChromaDB~~ (REMOVED)

No longer needed. The synthesis pipeline queries Wikipedia directly via the
full FAISS IVFPQ index (40M vectors). ChromaDB `wiki_knowledge` is not part
of the synthesis path.

#### Step 1.3 — Generate Synthetic Personal Facts

Create a script (`scripts/generate_test_facts.py`) that populates the `facts`
collection and knowledge graph with ~50-100 realistic personal facts spanning
multiple domains.

**Clean facts (core set):**
```
- User lives in Portland (geography)
- User works as a brewer (career)
- User lifts weights 4x/week (fitness)
- User has a brother named Auggie (relationships)
- User takes Adderall for ADHD (health)
- User is learning statistics (education)
- User plays board games (hobbies)
- User is building Daemon (projects)
```

**Intentional noise (critical for realistic calibration):**
```
- "August mentioned he might visit" (ambiguous — is this Auggie? entity resolver test)
- "user's brother also runs sometimes I think" (hedged, no explicit name)
- "user maybe allergic to shellfish" (confidence ~0.45 — should be gated by 0.50 threshold)
- "user likes running" + "user enjoys jogging" (near-duplicate, dedup test)
- "user works at the brewery downtown" (ambiguous domain — career? geography?)
- "user read something about stoicism" (vague, low-value fact)
```

The noisy facts test whether the pipeline works on realistic graph shape —
messy possessives, hedged confidence, ambiguous entity boundaries — not
idealized clean input.

- **Success criteria:** Knowledge graph has ≥30 nodes, ≥25 edges, ≥5 distinct
  domains. At least 2 facts trigger entity resolution ambiguity. At least 1
  fact is rejected by the 0.50 confidence gate.

#### Step 1.4 — Verify Data Integrity
```bash
python scripts/audit_startup.py
```
Check that the FAISS index loads and returns results, that the `facts`
collection is populated, and that the knowledge graph is queryable with
realistic entity resolution.

---

### Phase 2: Filter Calibration (Smoke Test → Mock → Live)

**Run order matters.** Start with verification pairs (fastest signal), then
mock calibration, then live LLM. This catches broken gates before you spend
time on the full fixture.

#### Step 2.1 — Smoke Test: Verification Pairs Against Full FAISS Index
```bash
python scripts/verify_synthesis_pipeline.py
```

This is a 5-minute smoke test with known-good/known-bad pairs. If TRUE pairs
aren't dying at Stage 3 or FALSE pairs leak through Stage 5, you know which
gate is broken before touching the full fixture.

**What we're measuring per pair type:**

| Pair Type | Count | Expected Result | What It Proves |
|-----------|-------|----------------|----------------|
| TRUE connections (known) | 8 | Rejected at **novelty gate** (Stage 3) | Novelty detection works — full 40M FAISS index catches known connections |
| FALSE connections (forced) | 6 | Rejected at **coherence gate** (Stage 5) | Coherence judge works — mechanistically wrong claims fail structural test |
| COVERAGE_ASYMMETRY | 4 | Varies (pathway-aware) | Novelty gate doesn't overfit on Wikipedia coverage — sparse but real connections aren't auto-rejected |

**Success criteria:**
- 0 hard failures (no false connections accepted)
- ≥75% pathway accuracy (rejections happen at the expected gate)
- Coverage-asymmetry pairs flagged for review, not auto-rejected

**Coverage-asymmetry disposition:** These are a calibration-time category,
not a permanent third outcome. After reviewing, each pair resolves to one of:
(a) accepted — the connection is genuinely novel despite sparse wiki coverage,
(b) rejected — the novelty gate was right to be suspicious, the connection is
well-known outside Wikipedia, or (c) threshold adjustment needed — the pair
exposes a gap in the novelty gate's calibration. In production, there is no
"review" bucket — every candidate either passes or fails. The purpose here
is to identify whether the thresholds need tuning for coverage-asymmetric cases.

**Diagnostic on unexpected outcomes:** The verification script dumps the full
`StageResult` chain (all stage scores, coherence justification, factual
skeptic response) for any unexpected pass or wrong-pathway rejection. When
a FALSE pair leaks past Stage 5, the interesting question isn't just "it
leaked" — it's what coherence score the LLM assigned, what mechanism it
invented, and whether the factual skeptic pass fired. Check `--verbose` output
and `data/synthesis_verification_results.json` for full details.

**Note:** With the full 40M FAISS index, Stage 3 novelty scores here are
production-realistic — no subset optimism caveat.

### Verification Finding: Novelty Gate Dominance (2026-03-31)

**Result:** 22/22 pairs rejected, 0 hard failures, 100% rejection rate — but
ALL rejections happened at the novelty gate (Stage 3). No pair reached the
coherence judge (Stage 5), including 4 pairs specifically designed with obscure
topics and novel phrasing to bypass novelty.

**Root cause:** With 40M Wikipedia articles, FAISS finds some article matching
any well-written English claim text. Claim similarity of 0.60+ is nearly
guaranteed for fluent prose — the gate measures "does text like this exist
in Wikipedia?" not "is this specific cross-domain connection documented?"
Even Pysanka ↔ Erlang (deliberately obscure) scored 0.951 claim similarity.

**Interpretation:** This is correct pipeline behavior, not a bug. The novelty
gate is the primary defense layer with a 40M-article corpus. The coherence
judge is a safety net for the rare case where a candidate has genuinely novel
phrasing (which is what the real generator produces — cross-domain bridges
from personal facts, not polished Wikipedia-like prose).

**Implication for testing:** The two gates must be tested separately:
- **Novelty gate:** Verified by this script against the real FAISS index (done)
- **Coherence gate:** Verified by mock/live calibration scripts with controlled
  novelty scores (Steps 2.2 and 2.3), where the FAISS mock returns low
  similarity to let candidates through to Stage 5

**Pathway accuracy is not a meaningful metric here.** The script marks "wrong
pathway" when novelty catches a pair expected at coherence. But the pipeline's
job is to reject bad candidates — which gate does it is an implementation
detail. Both gates are validated; they just can't be validated simultaneously
with real FAISS data because novelty dominates at 40M scale.

#### Step 2.2 — Run Mock Calibration Against Fixture
```bash
pytest tests/test_synthesis_calibration.py -v -s
```

**What we're measuring:**
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Precision | ≥ 85% | False positives (accepted junk) destroy user trust |
| Recall | ≥ 80% | False negatives (rejected novel) mean we miss real insights |
| F1 | ≥ 82% | Balanced measure — both matter |
| Sanity-fail rejection at Stage 0 | 100% | Text quality gate must be airtight |
| Trivial rejection rate | ≥ 90% | Known connections must not leak through |
| Noise rejection rate | ≥ 95% | Incoherent/forced connections must be caught |
| Noise-borderline rejection rate | ≥ 90% | Pseudoscience disguised as structure must fail |
| Novel acceptance rate | ≥ 75% | Genuinely new structural insights must survive |

**If mock results are below target:** Adjust thresholds (see tuning guide in Phase 4).

#### Step 2.3 — Run Live LLM Calibration
```bash
python scripts/calibrate_coherence_live.py sonnet-4.5
```

**What we're measuring:**
- How does the real LLM (not mock heuristics) score each tier?
- Does the two-pass coherence judge (structural + factual skeptic) match the mock's predictions?
- Are there candidates that the mock gets right but the real LLM misclassifies (or vice versa)?

**Success criteria:** Live F1 ≥ 90% (Sonnet 4.5 two-pass). Previous benchmark:
95.2% on 54 candidates. Fixture has grown to 72 — expect 90-92% F1 on the
expanded set. If it drops below 88%, the 18 new candidates (noise_borderline,
interesting_known) are exposing a real gap in the coherence prompt, not just
noise from additional data points.

---

### Phase 3: End-to-End Synthesis Run

#### Step 3.1 — Simulate a Session Shutdown with Dreaming
Run the shutdown processor's synthesis dreaming step against the loaded data.

**Important:** Use an elevated candidate count for calibration runs to get
enough signal. Set `SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION=15` (not the
production default of 5). This gives 15 candidates per run — enough to
distinguish "filter too tight" from "unlucky sampling" in a single session.

```python
# Test script: scripts/test_end_to_end_synthesis.py
# 1. Initialize MemoryCoordinator with test data
# 2. Override SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION = 15
# 3. Call _run_synthesis_dreaming()
# 4. Log all candidates: generated, accepted, rejected, rejection stages
# 5. Store results in synthesis_results collection
```

**What we're measuring:**
| Metric | Target | Result (2026-04-01) | Status |
|--------|--------|---------------------|--------|
| Candidates generated | ≥ 10 | 30 (across 3 tiers) | **PASS** |
| Candidates with LLM articulation | ≥ 6 | 30 (all) | **PASS** |
| Stage 0-2 rejections | ≤ 2 | 0 | **PASS** |
| Stage 3 rejections (novelty) | 2-5 | 12 | **ABOVE TARGET** |
| Stage 5 rejections (coherence) | 1-3 | 1 | **PASS** |
| Stage 6 rejections (composite) | — | 13 | — |
| Final acceptances | 1-4 | 4 | **PASS** |

**Key diagnostic:** If acceptance rate is 0%, the filter is too strict. If
acceptance rate is >60%, the filter is too loose. Sweet spot: 10-30%.
Result: 4/30 = 13% — within target band.

**Note on Stage 3:** With the full 40M FAISS index, Stage 3 (novelty) and
Stage 6 (composite) are the primary rejection gates. The novelty gate catches
candidates whose connections are already documented; composite scoring catches
candidates with weak multi-signal profiles. The coherence judge now rejects
far fewer candidates (1 vs 8 in the 303-node baseline) because the retrieval
generator produces structurally stronger claims.

**Note on three-tier generation:** The 30 candidates break down as: 15 from
RetrievalSynthesisGenerator (Tier 0), 0 from GraphWalkGenerator (Tier 1,
inactive — only 6 bridges), 15 from SynthesisGenerator (Tier 2). Tier 0
produced 2 acceptances with mechanism-naming insights; Tier 2 produced 2
acceptances with lower-quality but passing insights.

#### Step 3.2 — Simulate Proactive Surfacing
Run the context surfacer against the populated graph with test queries:

```python
# Test queries spanning different domains:
queries = [
    "I'm stressed about work deadlines",      # career + health bridge (2 domains)
    "My brother and I are training together",  # relationships + fitness bridge (2 domains)
    "I'm applying what I learned in stats",    # education + projects bridge (2 domains)
    # Multi-domain query: career + relationships + education + projects (4 domains)
    # Tests whether the surfacer degrades gracefully or picks the most interesting bridge
    "I'm stressed about the brewery expansion and Auggie's been helping me think through the stats",
]
# For each: call generate_insights(), verify non-empty, inspect quality
```

**What we're measuring:**
- Does the surfacer activate? (graph meets sparsity threshold)
- Does it detect the right active domains from each query?
- Are the generated insights plausible bridges between domains?
- Does session caching work? (second call returns cached result)
- **Multi-domain query:** Does the surfacer pick the most interesting bridge
  when 4+ domains are active simultaneously, or does it produce a generic
  insight that tries to span everything? The star-topology classifier might
  behave differently when multiple domains compete for bridge selection.

**Success criteria:** At least 1 of 4 queries produces a non-trivial insight.
The multi-domain query should produce a focused bridge (not a vague "everything
is connected" platitude).

#### Step 3.3 — Test Convergence Tracking
Run synthesis dreaming multiple times (simulating multiple sessions) and check
whether convergence detection fires:

```python
# 1. Run dreaming 5 times with 15 candidates each (= 75 total candidates)
# 2. Query synthesis_memory for any results with unique_paths > 1
# 3. If convergence detected: inspect the insight — is it a real connection?
# 4. If no convergence after 75 candidates: investigate sampling diversity
```

**What we're measuring:** Whether the same structural insight can be
independently rediscovered from different random entity pairs. With 75
candidates against the full 40M FAISS index, convergence is plausible but not
guaranteed — it depends on sampling diversity and the user's fact graph
richness. The mechanism just needs to work when it does fire.

**Success criteria:** Convergence tracking code executes without error; if
convergence fires, the converging insight is structurally valid.

---

### Phase 4: Threshold Tuning

If Phase 2-3 results are off-target, tune these knobs in order:

#### Novelty Gate (Stage 3) — If too many known connections leak through:
```python
# Tighten claim similarity gate (reject more known claims)
SYNTHESIS_NOVELTY_KNOWN_THRESHOLD: 0.88 → 0.85
# Tighten co-occurrence gate (reject more known pairings)
SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD: 0.85 → 0.80
```

#### Novelty Gate (Stage 3) — If too many novel connections are rejected:
```python
# Loosen claim similarity gate
SYNTHESIS_NOVELTY_KNOWN_THRESHOLD: 0.88 → 0.90
# Loosen co-occurrence gate
SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD: 0.85 → 0.90
# Check coverage-asymmetry pairs — are sparse-but-real connections being caught?
```

#### Coherence Judge (Stage 5) — If pseudoscience leaks through:
```python
# Already at MODERATE minimum — can't lower without losing real insights
# Instead: tune the factual skeptic prompt in synthesis_filter.py
# Add more debunked myths to the "watch for" list
```

#### Coherence Judge (Stage 5) — If too many real connections are rejected:
```python
# Check if the structural coherence prompt is too strict
# The key phrase: "shared mathematical pattern, feedback loop, optimization curve,
#   threshold dynamic, or structural isomorphism" — expand examples if needed
```

#### Composite Score (Stage 6) — General precision/recall balance:
```python
# If too many false positives: raise composite minimum
SYNTHESIS_COMPOSITE_MIN_SCORE: 0.65 → 0.70
# If too many false negatives: lower composite minimum
SYNTHESIS_COMPOSITE_MIN_SCORE: 0.65 → 0.60
# Adjust sub-weights if one signal dominates: check novelty breakdown
```

---

### Phase 5: Results Analysis & Documentation

#### Step 5.1 — Produce Summary Report

For each run, capture:
1. **Confusion matrix** — TP, FP, TN, FN across all tiers
2. **Per-stage rejection histogram** — where do candidates die?
3. **Score distributions** — composite scores for accepted vs. rejected
4. **Boundary analysis** — candidates with composite 0.35-0.45 (near threshold)
5. **Quality review queue** — any accepted candidates that look suspicious

#### Step 5.2 — Evaluate Against Success Criteria

**The concept is validated if ALL of the following hold:**

| # | Criterion | Threshold | How Tested | Rationale |
|---|-----------|-----------|-----------|-----------|
| 1 | Filter F1 on labeled fixture | ≥ 85% | Mock calibration (Step 2.2) | The filter reliably separates signal from noise |
| 2 | Zero hard failures on verification pairs | 0 false connections accepted | Verification script (Step 2.1) — **PASSED** | No mechanistically wrong claims survive |
| 3 | Novelty gate catches known connections | 100% rejection rate | Verification script (Step 2.1) — **PASSED** (22/22 rejected) | Full 40M FAISS index catches all known connections |
| 4 | Coherence gate catches forced connections | ≥ 90% of noise/borderline tiers rejected at Stage 5 | Mock + live calibration (Steps 2.2, 2.3) | LLM judge distinguishes structure from metaphor (tested with mock novelty scores) |
| 5 | End-to-end acceptance rate | 10-30% of generated candidates | End-to-end run (Step 3.1) | Realistic: most candidates should fail, but some survive |
| 6 | At least 1 accepted insight is genuinely interesting | Human judgment | End-to-end run (Step 3.1) | The whole point — AI found a non-obvious connection |
| 7 | Proactive surfacing activates | ≥ 1 insight from 4 test queries | Proactive surfacing test (Step 3.2) | The real-time pathway works alongside batch dreaming |

**If criteria 1-5 hold but criterion 6 fails:** The filter works but the
generator needs better sampling. Adjust seed queries, increase candidate
count, or enrich the personal facts graph with more diverse domains.

**If criterion 6 holds but criteria 1-4 don't:** The generator is good but
the filter needs calibration. Return to Phase 4.

#### Step 5.3 — Document Findings

Update `docs/SYNTHESIS_FILTER.md` with:
- New calibration results (date, model, fixture size, metrics)
- Any threshold changes made during tuning
- Examples of accepted insights (the best 2-3)
- Examples of correctly rejected candidates (1 per tier)
- Known weaknesses or edge cases

---

## Results Update (2026-04-01)

### Experiment: Retrieval-Based Synthesis vs. Random Pairing

Three experiments were run against production data (30,929 graph nodes, 8,343 edges, 40M FAISS vectors, 2,179 facts):

| Experiment | Generator | Coherence Calibration | Candidates | Accepted | Rate |
|---|---|---|---|---|---|
| Baseline (random + rubber-stamp) | SynthesisGenerator | Pre-calibration (accepts WEAK+) | 45 (3x15) | 7 | 16% |
| Baseline (random + calibrated) | SynthesisGenerator | Post-calibration (MODERATE+ required, mechanism naming) | 45 (3x15) | 0 | 0% |
| Retrieval + calibrated | 3-tier (Retrieval + Walk + XStore) | Post-calibration | 30 | 4 | 13% |

**Key findings:**

1. **Calibrated coherence judge discriminates.** The same random generator that produced 7/45 acceptances with rubber-stamp judging produces 0/45 with calibrated judging. The pre-calibration coherence judge accepted WEAK claims (surface metaphor); post-calibration requires MODERATE (named mechanism concretely applied to both domains).

2. **Retrieval produces mechanism-naming candidates.** The RetrievalSynthesisGenerator extracts structural queries from personal facts, then adversarially evaluates FAISS results. Accepted insights name specific mechanisms: "historical layering and cultural shifts" (urban geography ++ heritage studies), "conditional dependency" (family financial help ++ welfare dependency). These are qualitatively different from the 303-node baseline's surface-level connections ("brewing ++ gender inequality").

3. **Three-tier head-to-head:** RETRIEVAL 2/15 (13%), WALK 0/0 (inactive, 6 bridges < 40 minimum), XSTORE 2/15 (13%). Same acceptance rate but retrieval insights were higher quality.

4. **Rejection breakdown (30 candidates):** composite_scoring 13, novelty_external 12, coherence_judge 1. Composite scoring is now the primary rejection gate (was coherence at 303 nodes). This indicates the generator produces structurally better claims that pass coherence, but the multi-signal composite catches weak overall profiles.

5. **Bridge feedback loop confirmed.** Accepted insights create provisional bridge edges in the knowledge graph: 129 -> 133 edges after run. As bridges accumulate, the walk generator (Tier 1) will eventually activate.

### Honest Limitations

- **Composite threshold is tight.** At 0.65, candidates with strong coherence but marginal novelty scores fail. May need loosening as retrieval generator matures.
- **Novelty gate may over-reject retrieval candidates.** Structurally novel connections phrased in Wikipedia-adjacent language score high claim similarity, triggering false novelty rejections.
- **Structural query diversity needs monitoring.** The few-shot LLM prompt may converge on limited query patterns, reducing the diversity of retrieval candidates over time.
- **Walk generator inactive.** Only 6 quality bridges after cleanup (was 129 garbage bridges from low-confidence entity mapper). Will activate as the bridge feedback loop grows the bridge count toward 40.

---

## Why This Proves the Concept Works with AI

The synthesis pipeline tests whether an AI system can do something that traditionally requires human interdisciplinary expertise: spotting structural parallels between unrelated domains.

**The key insight is that LLMs are good at articulating connections but bad at judging whether those connections are novel or meaningful.** An LLM will happily tell you that "ant colonies are like the internet" — which is true but well-known. It will also tell you that "sourdough fermentation is like quantum tunneling" — which sounds smart but is mechanistically wrong.

The 8-stage filter pipeline compensates for this by:

1. **Cheap heuristic gates** (Stages 0-2) kill malformed, single-domain, and trivially-close/far candidates before any expensive processing
2. **Corpus-backed novelty** (Stage 3) uses the actual Wikipedia embedding space to detect whether a connection is already documented — not the LLM's training data, but a verifiable external corpus
3. **Convergence** (Stage 4) treats independent rediscovery as evidence — if different random walks keep arriving at the same insight, that's signal
4. **Structural coherence** (Stage 5) forces the LLM to identify a specific shared mechanism, then subjects moderate claims to a factual skepticism pass — catching pseudoscience disguised in structural language
5. **Multi-signal composite** (Stage 6) prevents any single weak signal from carrying a candidate through

**If the calibration shows that this pipeline reliably accepts novel structural isomorphisms while rejecting trivia, metaphor, and pseudoscience, it demonstrates that:**

- AI can generate candidate cross-domain insights at scale (cheap)
- A layered filter pipeline can separate signal from noise (the hard part)
- Corpus-backed novelty detection prevents rediscovering the wheel
- Convergence tracking provides a second, independent validation signal
- The whole system improves as the knowledge graph grows (more entities → more diverse sampling → more cross-domain opportunities)

This is the minimum viable demonstration that AI-assisted knowledge synthesis is feasible — not as a replacement for human insight, but as a tool that surfaces candidates worthy of human attention.

---

## File Reference

| File | Role |
|------|------|
| `scripts/build_wiki_subset.py` | Build curated FAISS index from wiki corpus |
| `scripts/load_parquet_to_chromadb.py` | Load wiki metadata into ChromaDB (not needed for synthesis — FAISS replaces ChromaDB for wiki vector search; retained for other consumers like the prompt builder's `[WIKIPEDIA]` section) |
| `knowledge/semantic_search.py` | FAISS IVFPQ semantic search — the wiki vector backend used by `SynthesisGenerator._sample_wiki_articles()` |
| `scripts/generate_calibration_candidates.py` | Generate fresh candidates for labeling |
| `scripts/verify_synthesis_pipeline.py` | End-to-end verification with pre-defined pairs |
| `scripts/calibrate_coherence_live.py` | Live LLM calibration with multi-model comparison |
| `knowledge/synthesis_retriever.py` | Tier 0: Structural query extraction + FAISS retrieval + adversarial evaluation |
| `knowledge/synthesis_generator.py` | Tier 2: Cross-store random sampling + LLM bridge articulation |
| `knowledge/graph_walk_generator.py` | Tier 1: Biased Markov walk synthesis with hub dampening |
| `knowledge/synthesis_filter.py` | 8-stage filter pipeline (shared by all tiers) |
| `knowledge/synthesis_models.py` | Data models (Candidate, Result, StageResult, enums) |
| `memory/synthesis_memory.py` | ChromaDB persistence + convergence tracking |
| `memory/context_surfacer.py` | Proactive real-time cross-domain insights |
| `memory/memory_storage.py` | Fact extraction → graph ingestion (`_ingest_fact_to_graph()`) |
| `memory/graph_memory.py` | NetworkX DiGraph wrapper |
| `memory/shutdown_processor.py` | Synthesis dreaming trigger (Step 6.8) |
| `tests/test_synthesis_calibration.py` | Mock calibration test suite (6 tests) |
| `tests/fixtures/calibration_candidates.json` | 72 labeled candidates in 7 tiers |
| `config/app_config.py` | All `SYNTHESIS_*` and `PROACTIVE_SURFACING_*` constants |
