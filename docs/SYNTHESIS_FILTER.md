# Synthesis Filter Pipeline

Operational guide for the knowledge synthesis system — candidate generation,
8-stage filtering, convergence tracking, and calibration.

For formal notation see `FORMAL_MODEL.md` Section 13. For config constants
see `QUICK_REFERENCE.md` under "Knowledge Synthesis Filter Pipeline."

---

## What Synthesis Does

Daemon's synthesis system discovers non-obvious connections between concepts
the user knows about (personal facts) and general knowledge (Wikipedia).
It runs as a "dreaming" step during session shutdown — after conversations
are stored but before the process exits.

**The core idea:** Sample entities from the user's fact store (ChromaDB)
and Wikipedia (FAISS index, 40M vectors), pair them across domains, use an
LLM to articulate a connection, then run the candidate through an 8-stage
filter that kills noise, rehashes, and pseudoscience. Only genuinely novel,
coherent, cross-domain insights survive.

Accepted insights are stored in the `synthesis_results` ChromaDB collection
and can be surfaced in future conversations. When the same insight is
independently rediscovered via a different graph path, convergence tracking
strengthens confidence in it.

**When it runs:** `shutdown_processor.py` Step 6.8, after thread extraction
and before graph save. Gated by `SYNTHESIS_GENERATOR_ENABLED` and graph
sparsity guard (`SYNTHESIS_GENERATOR_MIN_GRAPH_NODES`). Additionally,
auto-halts if the human audit FP rate exceeds `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD`
(requires at least `SYNTHESIS_AUDIT_MIN_GRADED` graded results).

---

## File Map

| File | Purpose |
|------|---------|
| `knowledge/synthesis_models.py` | Data models: SynthesisCandidate, SynthesisResult, StageResult, CoherenceLevel, CandidateStatus |
| `knowledge/synthesis_retriever.py` | RetrievalSynthesisGenerator: structural query extraction + FAISS search + adversarial evaluation (Tier 0) |
| `knowledge/graph_walk_generator.py` | GraphWalkGenerator: biased Markov walks with hub dampening + cross-domain constraint (Tier 1) |
| `knowledge/synthesis_generator.py` | SynthesisGenerator: cross-store sampling (ChromaDB facts + FAISS wiki) + LLM bridge articulation (Tier 2) |
| `knowledge/synthesis_filter.py` | 8-stage filter pipeline + FAISS wiki novelty checks + template patterns + helpers |
| `memory/synthesis_memory.py` | ChromaDB persistence (synthesis_results collection) + convergence tracking + provisional bridge creation + audit queue (human grading, FP/FN review) |
| `config/app_config.py` | All `SYNTHESIS_*` constants (filter + generator + retrieval) |
| `memory/shutdown_processor.py` | Integration point — runs all three generators then filter at shutdown |
| `tests/test_synthesis_calibration.py` | Mock calibration suite (6 tests) |
| `tests/unit/test_synthesis_generator.py` | Generator unit tests (18 tests) |
| `tests/unit/test_synthesis_audit.py` | Audit queue unit tests (27 tests) |
| `tests/unit/test_graph_walk_generator.py` | Graph walk generator tests (38 tests) |
| `tests/fixtures/calibration_candidates.json` | 72 labeled candidates in 7 tiers |
| `scripts/calibrate_coherence_live.py` | Live LLM calibration with multi-model comparison |
| `scripts/test_end_to_end_synthesis.py` | E2E test: all three generators head-to-head with per-generator acceptance stats |

---

## Data Models

### SynthesisCandidate (input to filter)

```python
concept_a: str              # e.g. "bone remodeling"
concept_b: str              # e.g. "database index optimization"
connection_claim: str       # LLM-articulated bridge statement
walk_path: List[str]        # graph walk node sequence (or synthetic [a, b])
source_domains: Set[str]    # e.g. {"science", "technology"}
endpoint_distance: float    # 0-1, graph shortest path or default 0.55
timestamp: datetime
path_hash: str              # SHA-256[:16] of walk_path, for dedup
```

### SynthesisResult (output from filter)

Accumulates signals through the pipeline:

```python
candidate: SynthesisCandidate
stage_results: List[StageResult]        # per-stage pass/fail + score + metadata
coherence_level: CoherenceLevel         # INVALID/WEAK/MODERATE/STRONG
coherence_justification: str            # full LLM response text
novelty_score_external: float           # 1 - claim_sim (from Stage 3)
novelty_score_internal: float           # 1 - synthesis_memory_sim (from Stage 4)
cooccurrence_similarity: float          # bare "A B" FAISS wiki similarity (from Stage 3)
template_similarity: float              # generic bridge pattern score (from Stage 3)
nearest_known_external: str             # closest wiki match text
nearest_known_internal: str             # closest synthesis memory match text
composite_score: float                  # final weighted score (from Stage 6)
status: CandidateStatus                 # PENDING → ACCEPTED or REJECTED
rejection_stage: Optional[str]          # which stage killed it
rejection_reason: str

# Convergence tracking
unique_paths: Set[str]                  # path_hashes that found this insight
unique_sources: Set[str]                # "concept_a|concept_b" pairs
convergence_strength: float             # |paths| * |sources|

# Audit fields [NEW 2026-04-01]
human_grade: Optional[str]              # TRUE_POSITIVE, FALSE_POSITIVE, or FALSE_NEGATIVE
graded_at: Optional[datetime]           # When the human graded it
grade_notes: str                        # Free-text grading notes
```

### CoherenceLevel

```python
INVALID  = 0.0    # factually wrong or pure wordplay
WEAK     = 0.33   # no shared mechanism, surface metaphor only
MODERATE = 0.66   # real shared structural pattern
STRONG   = 1.0    # predictive cross-domain connection
```

---

## Candidate Generation

Three generators produce candidates in parallel at shutdown, each with
independent quotas. All emit `SynthesisCandidate` objects for the same
8-stage filter pipeline.

### Tier 0: RetrievalSynthesisGenerator (Retrieval-Based)

`knowledge/synthesis_retriever.py` — highest-quality candidates via
structural query extraction and FAISS semantic search.

**Pipeline:**
1. LLM structural query extraction (few-shot): given a personal fact,
   generate a query describing the *structural pattern* (e.g. "systems
   where load-bearing stress drives adaptive restructuring").
2. FAISS semantic search against 40M Wikipedia vectors using the
   structural query. Top-K results filtered by minimum similarity.
3. Adversarial evaluation: LLM judges whether the connection is
   genuinely structural or surface-level.
4. Package survivors as `SynthesisCandidate` objects.

**Config:** `SYNTHESIS_RETRIEVAL_ENABLED`, `SYNTHESIS_STRUCTURAL_QUERY_MAX_TOKENS` (default 100),
`SYNTHESIS_RETRIEVAL_K` (default 5), `SYNTHESIS_RETRIEVAL_MIN_SIMILARITY` (default 0.25).

### Tier 1: GraphWalkGenerator (Wikidata Graph Walks)

`knowledge/graph_walk_generator.py` — biased Markov random walks on the
unified personal+wikidata graph. Node2Vec-style return bias (2.0x toward
personal nodes when in wikidata territory).

**Improvements:**
- **Hub dampening**: Log-scale penalty for nodes with degree >
  `GRAPH_WALK_HUB_DEGREE_THRESHOLD` (default 15). Prevents walks from
  being dominated by highly-connected hub nodes.
- **Cross-domain walk constraint**: Walks must touch at least
  `GRAPH_WALK_MIN_DOMAINS` (default 2) distinct domain categories.
  Single-domain walks are discarded.

**Config:** `GRAPH_WALK_ENABLED`, `GRAPH_WALK_MIN_BRIDGE_EDGES` (default 40),
`GRAPH_WALK_HUB_DEGREE_THRESHOLD` (default 15), `GRAPH_WALK_MIN_DOMAINS` (default 2).

### Tier 2: SynthesisGenerator (Cross-Store Sampling)

`knowledge/synthesis_generator.py` — the original generator. Cross-store
sampling from ChromaDB facts + FAISS wiki.

**Pipeline:**
1. Sample 6 random query seeds from `_PERSONAL_QUERY_SEEDS` (e.g. "my family
   and friends", "my work and career") → query `facts` collection
2. Sample 6 random query seeds from `_WIKI_QUERY_SEEDS` (e.g. "scientific
   discovery breakthrough", "biological mechanism adaptation") → FAISS
   `semantic_search_with_neighbors()` against 40M Wikipedia vectors (IVFPQ
   index). Results normalized to match the ChromaDB dict shape.
3. Form cross-domain pairs: shuffle, deduplicate by concept name, skip
   same-domain pairs. Domain classification reuses `categorize_relation()`
   for personal facts and keyword heuristics for wiki articles.
4. LLM bridge articulation: parallel calls (semaphore-controlled) with
   `BRIDGE_PROMPT`. The LLM either articulates a specific connection or
   responds `NO_CONNECTION`.
5. Package survivors as `SynthesisCandidate` objects with synthetic walk
   paths and graph-based endpoint distances.

**Namespace resolution:** `_resolve_to_graph()` uses a 4-strategy fallback:
EntityResolver → direct ID → slug form → display_name index (30K entries).
`_compute_endpoint_distance()` returns 0.85 for both-resolved-no-path
instead of the old 0.55 fallback.

**Config:** `SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION` (default 5),
`SYNTHESIS_GENERATOR_LLM_CONCURRENCY` (default 5).

### Shutdown Orchestration

`_run_synthesis_dreaming()` in `shutdown_processor.py` runs all three
generators with independent quotas. Each generator fills its allocation,
then all candidates pass through the same `SynthesisFilter` pipeline.

---

## Data Flow

```
Parallel Candidate Generation (shutdown step 6.8):
  ├── [Tier 0] RetrievalSynthesisGenerator
  │     ├── Structural query extraction (few-shot LLM)
  │     ├── FAISS semantic search (40M vectors)
  │     └── Adversarial evaluation → SynthesisCandidate[]
  ├── [Tier 1] GraphWalkGenerator (if bridges >= 40)
  │     ├── Biased Markov walks (hub-dampened, cross-domain)
  │     └── Walk narration (LLM) → SynthesisCandidate[]
  └── [Tier 2] SynthesisGenerator
        ├── Sample facts (ChromaDB) + wiki articles (FAISS, 40M vectors)
        ├── Form cross-domain pairs
        ├── LLM bridge articulation (parallel, semaphore)
        └── Package as SynthesisCandidate[]
              │
              ▼ (all generators feed the same pipeline)
SynthesisFilter.process_candidate(candidate)
  ├── Stage 0: Text Sanity          (~0ms, regex)
  ├── Stage 1: Domain Crossing      (~1ms, metadata)
  ├── Stage 2: Semantic Distance    (~5ms, embedding)
  ├── Stage 3: Novelty External     (~15ms, 2 FAISS wiki searches + regex)
  │     ├── Sub-check 1: Claim similarity
  │     ├── Sub-check 2: Co-occurrence
  │     └── Sub-check 3: Template specificity
  ├── Stage 4: Novelty Internal     (~10ms, synthesis memory)
  ├── Stage 5: Coherence Judge      (~1-4s, 1-2 LLM calls)
  │     ├── Pass 1: Structural coherence
  │     └── Pass 2: Factual skeptic (MODERATE only)
  ├── Stage 6: Composite Scoring    (~0ms, arithmetic)
  └── Stage 7: Storage              (~10ms, ChromaDB write)
        │
        ▼
SynthesisMemory.store_result(result)
  ├── New insight → store in synthesis_results collection
  └── Existing insight, new path → update convergence tracking
```

Short-circuit: any stage failure immediately rejects the candidate. Stages
are ordered cheap-to-expensive so most noise dies before the LLM fires.

---

## Stage-by-Stage Walkthrough

### Stage 0: Text Sanity

**Purpose:** Kill malformed, empty, or repetitive claims before anything else.

**Checks:**
- Token count >= 10 (`SYNTHESIS_MIN_TOKEN_LENGTH`)
- At least one verb (regex + common verb set)
- Repetition ratio < 0.5 (`SYNTHESIS_MAX_REPETITION_RATIO`)

**Example rejection:**
```
Claim: "trees money growth economy"
Reason: No verbs detected
```

### Stage 1: Domain Crossing

**Purpose:** Reject candidates that don't actually cross domain boundaries.

**Check:** `len(source_domains) >= 2` (`SYNTHESIS_MIN_DOMAINS`)

**Score:** `min(domain_count / 4.0, 1.0)` — 4+ domains scores 1.0.

### Stage 2: Semantic Distance

**Purpose:** Filter endpoints that are trivially close (same topic) or
nonsensically far (no embedding relationship).

**Range:** `[0.20, 0.90]` (`SYNTHESIS_DISTANCE_MIN`, `SYNTHESIS_DISTANCE_MAX`)

**Score:** Peaks at midpoint (0.55), drops toward edges.

**Retrieval-aware scoring:** For candidates from `RetrievalSynthesisGenerator`,
distance scoring is inverted — low distance (high semantic similarity) is
*good*, because the retrieval pipeline already selected structurally relevant
results. Standard generators penalize low distance as "trivially close."

**Example rejection:**
```
Concepts: "socks" <-> "dark matter"
Distance: 0.95 > 0.90 (nonsensical)
```

### Stage 3: Novelty External

Three sub-checks, each targeting a different failure mode.

#### Sub-check 1: Claim Similarity

Searches the FAISS wiki index (40M vectors) for the **full articulated
claim** via `semantic_search_with_neighbors()`. Catches direct rehashes.
Similarity scores are cosine similarity (0-1) returned directly from FAISS,
extracted by `_extract_faiss_similarity()`.

**Hard gate:** `claim_sim > 0.60` (`SYNTHESIS_NOVELTY_KNOWN_THRESHOLD`, IVFPQ-calibrated)

**Retrieval-aware skip:** For candidates from `RetrievalSynthesisGenerator`,
the claim-similarity sub-check is skipped. Retrieval candidates derive their
claims from wiki content by design, so high claim similarity is expected and
does not indicate a rehash.

**Example rejection:**
```
Claim: "Regular exercise increases serotonin production, improving mood"
Wiki match: "Exercise and Serotonin" article (similarity=0.90)
Reason: Claim already known
```

#### Sub-check 2: Co-occurrence

Searches the FAISS wiki index for the **bare concept conjunction**
`"concept_a concept_b"` via `semantic_search_with_neighbors()`. Catches the
"known connection, novel phrasing" failure mode — when the claim embeds far
from wiki (novel sentence) but the concepts already appear together in
literature.

**Hard gate:** `cooccurrence_sim > 0.85` (`SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD`, 40M-scale recalibrated)

**Example:** "Bacterial quorum sensing parallels voting theory" — the claim
itself is novel text, but searching `"quorum sensing voting theory"` hits
survey articles discussing exactly this analogy.

#### Sub-check 3: Template Specificity

Regex-based detection of vacuous bridge language. Catches the "novel
connection, vacuous claim" failure mode — coherent and embedding-novel but
substantively empty.

**Patterns matched** (`_GENERIC_TEMPLATES`):
```
"both involve/require/depend on/use/exhibit..."
"share structural/fundamental/deep similarities..."
"operates on similar/the same/analogous principles..."
"is an example/instance/case/form of..."
"mirrors/echoes/reflects the same/similar..."
"both are/represent forms/types/examples of..."
"just as X, so too/similarly/likewise..."
"at its core is really/essentially/fundamentally..."
"can be seen/viewed/understood as a form/instance of..."
```

**Generic tokens** (`_GENERIC_TOKENS`): systems, processes, principles,
dynamics, mechanisms, fundamental, inherently, essentially, paradigm,
parallels, interconnected, holistic, synergy, synergistic.

**Scoring:** `max(min(template_matches/2, 1.0), min(generic_tokens/4, 1.0) * 0.7)`.
Both sub-scores clamped to 1.0 before the final max. No hard gate — feeds into
composite via specificity signal.

**Example:**
```
Claim: "Both involve feedback loops and share structural similarities"
Template matches: 2 ("both involve", "share structural similarities")
Score: 1.0 (max vacuous)
```

### Stage 4: Novelty Internal

Checks synthesis memory for existing insights. Key behavior: same insight
via a **different path** doesn't reject — it updates convergence tracking.

**Convergence:** When `unique_paths >= 3` AND `unique_sources >= 2`, the
result promotes to `CONVERGING` status. Independent rediscovery from
different graph walks is a strong signal.

### Stage 5: Coherence Judge (Two-Pass LLM)

#### Pass 1: Structural Coherence

Distinguishes structural isomorphisms from loose analogies.

**System prompt:**
```
You evaluate cross-domain knowledge connections. Focus on whether a shared
structural pattern genuinely exists, not on whether the domains seem related.
Be skeptical of vague claims but generous toward specific structural parallels.
```

**User prompt:**
```
Evaluate this claimed connection between two concepts.

Concept A: {concept_a}
Concept B: {concept_b}
Claimed connection: {claim}

These concepts are from different domains — that is expected and desired.
Your job is to evaluate the SPECIFIC MECHANISM or SHARED STRUCTURE described,
not whether cross-domain connections are valid in general.

Key distinction: A connection that identifies a shared mathematical pattern,
feedback loop, optimization curve, threshold dynamic, or structural
isomorphism across domains is MODERATE or STRONG — even if the two systems
operate through different physical substrates. Only rate WEAK if the
connection is pure metaphor with no identifiable shared process.

First, identify the specific mechanism or structural pattern claimed
(e.g., negative feedback, diminishing returns, threshold collapse,
competitive selection).
Then, write one sentence on the strongest reason it might be WRONG.
Then, write one sentence on what makes it genuinely insightful.

Finally, choose exactly ONE rating:
INVALID - Factually wrong, or pure wordplay with no shared process
WEAK    - No specific shared mechanism; surface-level metaphor only
MODERATE - Real shared structural pattern, even through different substrates
STRONG  - Compelling connection; understanding one system predicts the other

Format:
Mechanism: <the specific shared pattern>
Against: <strongest criticism>
For: <what makes it insightful>
Rating: <INVALID|WEAK|MODERATE|STRONG>
```

**Gate:** `coherence_level >= MODERATE` (`SYNTHESIS_COHERENCE_MIN_LEVEL`)

**Model:** `SYNTHESIS_COHERENCE_MODEL` (default: `sonnet-4.5`). Calibrated
against GPT-4o-mini — Sonnet 4.5 achieves higher F1 on the fixture set.

**Recalibrated prompt standards:**
- **WEAK**: No mechanism named, vague bridging language ("both involve
  feedback loops" without specifying *which* feedback loops).
- **MODERATE**: Names a real mechanism and applies it concretely to both
  domains. Generality is not a flaw if application is concrete.
- Additional generic bridge templates added to catch vacuous claims that
  previously slipped through as MODERATE.

**LLM parameters:** `max_tokens=400` (raised from 250), `temperature=0.1`

#### Pass 2: Factual Skeptic

Fires **only on MODERATE** results from Pass 1. STRONG is confident enough
to skip; WEAK/INVALID already rejected.

Targets pseudoscience wrapped in structural language. Binary PASS/FAIL.
Simplification is acceptable — only provably wrong claims fail.

**Prompt:**
```
A cross-domain connection claims: {claim}

Check ONLY whether the domain-specific facts are wrong or based on
debunked science.

Examples of what FAILS:
- 'Left-brain people are more logical' (debunked lateralization myth)
- 'Mirror neurons cause empathy' (oversimplified; function is disputed)
- 'Mozart effect improves intelligence' (debunked)
- Claiming a specific neural pathway, enzyme, or mechanism that doesn't exist

Examples of what PASSES:
- Feedback loops that genuinely exist in both domains, even if simplified
- Structural parallels where both sides are real phenomena
- Using well-established concepts (Wolff's law, PID control) correctly

IMPORTANT: Simplification is expected. Only flag claims where a domain
expert would say 'that mechanism is wrong or doesn't exist,' NOT 'that's
a simplification.'

Respond:
PASS - Domain facts are real (even if simplified)
FAIL - A specific claim is factually wrong or based on debunked science

One sentence of reasoning, then PASS or FAIL on its own line.
```

**On FAIL:** Coherence downgraded to WEAK, candidate rejected.

**LLM parameters:** `max_tokens=150`, `temperature=0.1`

### Stage 6: Composite Scoring

Combines all prior signals into a single score.

**Top-level composite:**
```
composite = 0.30 * coherence + 0.40 * novelty + 0.15 * distance + 0.15 * structural
```

**Novelty is itself a 4-signal composite:**
```
novelty = 0.25 * (1 - claim_sim)           # claim novelty
        + 0.30 * (1 - cooccurrence_sim)     # co-occurrence novelty
        + 0.25 * (1 - template_sim)         # specificity
        + 0.20 * internal_novelty           # from synthesis memory
```

**Rationale for novelty weights:**
- Co-occurrence (0.30) — heaviest single weight because it most directly
  answers "is this relationship already documented?"
- Claim novelty (0.25) — still catches direct rehashes
- Specificity (0.25) — catches vacuous platitudes that pass other checks
- Internal (0.20) — lower weight because convergence from multiple paths
  is actually a positive signal

**Gate:** `composite >= 0.65` (`SYNTHESIS_COMPOSITE_MIN_SCORE`)

### Stage 7: Storage + Provisional Bridge Creation

Accepted results stored in `synthesis_results` ChromaDB collection.
If a similar insight already exists, convergence tracking is updated instead
of creating a duplicate.

**Provisional bridge creation:** When `SYNTHESIS_BRIDGE_ON_ACCEPT` is enabled,
accepted candidates also create a graph edge between their endpoint entities
via `SynthesisMemory.create_bridge_edge()`. The edge is created with
`weight=0.0` and `metadata.status="provisional"`, using the relation type
from `SYNTHESIS_BRIDGE_RELATION` (default: `"structural_parallel"`).
Provisional bridges mature when rediscovered: subsequent `add_relation()`
calls on the same edge increment the weight, eventually promoting the
bridge from provisional to established.

This requires `SynthesisFilter.__init__` to accept `graph_memory` and
`entity_resolver` parameters.

---

## Concrete Example: One Candidate Through All Stages

**Candidate:**
```
concept_a: "bone remodeling under stress"
concept_b: "database index optimization"
claim: "Wolff's law in bone biology — where repeated mechanical stress
  causes osteoclasts to remove unstressed bone and osteoblasts to
  reinforce loaded regions — mirrors how database engines continuously
  restructure B-tree indices by splitting hot nodes and merging cold
  ones based on query load patterns."
source_domains: {"science", "technology"}
endpoint_distance: 0.55
```

**Stage 0 — Text Sanity:** PASS
- Token count: 42 (>= 10)
- Verbs found: "causes", "remove", "reinforce", "mirrors", "restructure"
- Repetition ratio: 0.12 (< 0.5)
- Score: 0.88

**Stage 1 — Domain Crossing:** PASS
- Domains: {"science", "technology"} → 2 domains (>= 2)
- Score: 0.50

**Stage 2 — Semantic Distance:** PASS
- Distance: 0.55 (within [0.20, 0.90])
- Score: 1.0 (at midpoint — peak score)

**Stage 3 — Novelty External:** PASS (FAISS wiki index, 40M vectors)
- Sub-check 1 (claim sim): 0.15 — claim text is novel (< 0.80 threshold)
- Sub-check 2 (co-occurrence): 0.10 — "bone remodeling database index" is
  not a documented pairing in FAISS wiki vectors (< 0.75 threshold)
- Sub-check 3 (template sim): 0.00 — no generic templates match; claim uses
  specific terms (Wolff's law, osteoclasts, B-tree, hot nodes)
- novelty_score_external: 0.85, cooccurrence_similarity: 0.10, template_similarity: 0.00

**Stage 4 — Novelty Internal:** PASS
- No similar insight in synthesis memory
- novelty_score_internal: 1.0

**Stage 5 — Coherence Judge:**
- Pass 1 (structural): LLM identifies "stress-driven structural remodeling"
  as the shared pattern. Rates MODERATE.
- Pass 2 (factual skeptic): Wolff's law is real, B-tree splitting is real.
  Verdict: PASS. Keeps MODERATE.
- coherence_level: MODERATE (0.66)

**Stage 6 — Composite Scoring:** PASS
```
novelty = 0.25 * 0.85 + 0.30 * 0.90 + 0.25 * 1.00 + 0.20 * 1.00
       = 0.2125 + 0.27 + 0.25 + 0.20 = 0.9325

composite = 0.30 * 0.66 + 0.40 * 0.9325 + 0.15 * 1.0 + 0.15 * 0.50
         = 0.198 + 0.373 + 0.15 + 0.075 = 0.796
```
- composite: 0.796 (>= 0.65 threshold) → PASS

**Stage 7 — Storage:** Stored in `synthesis_results` collection. New insight,
no convergence update needed.

**Result:** ACCEPTED with composite 0.796.

---

## Calibration

### Fixture Set

`tests/fixtures/calibration_candidates.json` — 72 labeled candidates in 7 tiers:

| Tier | Count | Expected | Purpose |
|------|-------|----------|---------|
| `sanity_fail` | 4 | rejected | Malformed text, should die at Stage 0 |
| `trivial` | 18 | rejected | Well-known connections (exercise↔mood, sleep↔cognition) |
| `noise` | 12 | rejected | Forced/incoherent (shoelaces↔game theory) |
| `noise_borderline` | 10 | rejected | Plausible-sounding pseudoscience (mirror neurons, Mozart effect) |
| `interesting_known` | 8 | rejected | Published cross-domain work (ant colonies↔TCP) |
| `interesting_novel` | 15 | accepted | Non-obvious AND new (bone remodeling↔index optimization) |
| `boundary` | 5 | diagnostic | Hard middle cases, no expected outcome |

### Running Calibration

**Mock tests** (fast, deterministic):
```bash
pytest tests/test_synthesis_calibration.py -v -s
```

**Live LLM tests** (real API calls, ~2-3 min):
```bash
# Single model
python scripts/calibrate_coherence_live.py sonnet-4.5

# Compare models
python scripts/calibrate_coherence_live.py sonnet-4.5 gpt-4o-mini
```

### Reading the Output

Key metrics:
- **Precision** — Of candidates accepted, how many should have been? Low precision = too many false positives (noise getting through)
- **Recall** — Of candidates that should be accepted, how many were? Low recall = too many false negatives (good insights rejected)
- **Per-tier rejection stages** — Shows which stage is doing the work. If noise dies at coherence_judge, the cheap stages aren't filtering enough.

### Adding Ground Truth

To add a new calibration candidate:
1. Add an entry to `calibration_candidates.json` with all required fields
2. Set `tier` to one of the 7 tiers
3. Set `expected_outcome` to `"accepted"` or `"rejected"`
4. Run calibration to verify it behaves as expected
5. If it doesn't, investigate which stage is making the wrong call

---

## Tuning Guide

### Precision Too Low (accepting bad insights)

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Noise getting through Stage 5 | Coherence judge too generous | Add noise examples to calibration fixture, check if prompt needs tightening |
| Pseudoscience passing | Factual skeptic not catching it | Add FAIL examples to skeptic prompt |
| Vacuous claims accepted | Template patterns missing a pattern | Add regex to `_GENERIC_TEMPLATES` |
| Known connections passing | Co-occurrence threshold too high | Lower `SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD` |
| Composite scores too high | Novelty weights too generous | Increase `SYNTHESIS_NOVELTY_W_COOCCURRENCE` weight |

### Recall Too Low (rejecting good insights)

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Good claims rated WEAK | Pass 1 prompt too strict | Check if the prompt's WEAK definition is catching structural isomorphisms |
| Factual skeptic rejecting valid claims | Skeptic too aggressive | Narrow the FAIL criteria in skeptic prompt |
| Template check penalizing specific claims | Regex too broad | Check `_GENERIC_TEMPLATES` for false matches on specific language |
| Composite too low | Novelty weights penalizing unfairly | Check which novelty signal is dragging the score down |
| Distance gate killing good pairs | Range too narrow | Widen `SYNTHESIS_DISTANCE_MIN` / `SYNTHESIS_DISTANCE_MAX` |

### Convergence Too Slow

- Increase `SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION` to generate more candidates per session
- Lower `SYNTHESIS_GENERATOR_MIN_GRAPH_NODES` to allow generation with sparser graphs
- Check that the graph has enough nodes across different domains

---

## Benchmark Results (2026-03-29)

Tested against 72 labeled candidates with mock FAISS wiki index + real LLM
coherence. Candidates that die before Stage 5 (sanity, novelty, distance)
use deterministic mocks. Stage 5 uses live API calls.

| Model | Precision | Recall | F1 | FP | FN |
|-------|-----------|--------|-----|----|----|
| Sonnet 4.5 (two-pass) | 90.9% | 100% | **95.2%** | 1 | 0 |
| GPT-4o-mini (two-pass) | 83.3% | 100% | 90.9% | 2 | 0 |

**Remaining FP (Sonnet 4.5):** butterfly effect↔stock market crashes —
"nonlinear feedback loops and sensitivity to initial conditions" describes
real math but misapplies deterministic chaos theory to stochastic markets.

**Prompt iteration history:**
- Original prompt (pre-rewrite): Sonnet F1=30.8%, 4o-mini F1=42.9%
- Rewritten prompt (structural focus): Sonnet F1=90.9%, 4o-mini F1=85.7%
- + Factual skeptic: Sonnet F1=95.2%, 4o-mini F1=90.9%

---

## Audit Queue (Human-in-the-Loop Grading)

The synthesis pipeline produces candidates automatically, but precision can
only be truly measured by human review. The audit queue provides a blind
grading workflow with auto-halt capability.

### How It Works

1. **Accepted results** are stored with `human_grade=None` (ungraded).
2. **Composite-rejected candidates** (those that pass all gates up to Stage 6
   but score below `SYNTHESIS_COMPOSITE_MIN_SCORE`) are stored via
   `SynthesisFilter.process_batch()` → `synthesis_memory.store_rejected_for_audit()`
   for false-negative (FN) review.
3. The **GUI "Synthesis" tab** presents a blind review queue: the grader sees
   the claim, concepts, and coherence level but grades without knowing the
   pipeline's original verdict.
4. Grades: `TRUE_POSITIVE` (genuinely novel insight), `FALSE_POSITIVE`
   (noise that slipped through), `FALSE_NEGATIVE` (good insight the pipeline
   wrongly rejected).
5. **Auto-halt**: At shutdown, before running synthesis dreaming,
   `shutdown_processor.py` checks `synthesis_memory.get_audit_stats()`.
   If `fp_rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` and
   `graded >= SYNTHESIS_AUDIT_MIN_GRADED`, synthesis is skipped for that
   session with a warning log. This prevents accumulating more bad output
   while the pipeline is miscalibrated.

### Audit Stats

`synthesis_memory.get_audit_stats()` returns:

```python
{
    "total": int,       # Total results in collection
    "graded": int,      # Results with a human grade
    "ungraded": int,    # Results awaiting review
    "tp": int,          # TRUE_POSITIVE count
    "fp": int,          # FALSE_POSITIVE count
    "fn": int,          # FALSE_NEGATIVE count
    "fp_rate": float,   # fp / graded (0.0 if none graded)
}
```

### Config

| Constant | Default | Purpose |
|----------|---------|---------|
| `SYNTHESIS_AUDIT_ENABLED` | `True` | Master toggle for audit queue |
| `SYNTHESIS_AUDIT_FP_HALT_THRESHOLD` | `0.50` | FP rate above which synthesis auto-halts |
| `SYNTHESIS_AUDIT_MIN_GRADED` | `10` | Minimum graded results before auto-halt activates |

YAML section: `synthesis_audit`

### Tests

27 unit tests in `tests/unit/test_synthesis_audit.py`.

---

## Config Reference

All constants live in `config/app_config.py` under `SYNTHESIS_CFG` and
`SYNTHESIS_GEN_CFG`. YAML path: `synthesis:` and `synthesis_generator:`.

### Filter Pipeline

| Constant | Default | Stage | Purpose |
|----------|---------|-------|---------|
| `SYNTHESIS_ENABLED` | `True` | — | Master toggle for entire pipeline |
| `SYNTHESIS_MIN_TOKEN_LENGTH` | `10` | 0 | Minimum tokens in claim |
| `SYNTHESIS_MAX_REPETITION_RATIO` | `0.5` | 0 | Max allowed token repetition |
| `SYNTHESIS_MIN_DOMAINS` | `2` | 1 | Minimum domain count |
| `SYNTHESIS_DISTANCE_MIN` | `0.20` | 2 | Minimum endpoint distance |
| `SYNTHESIS_DISTANCE_MAX` | `0.90` | 2 | Maximum endpoint distance |
| `SYNTHESIS_NOVELTY_KNOWN_THRESHOLD` | `0.88` | 3 | Claim sim hard gate — near-verbatim rehashes only |
| `SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD` | `0.70` | 3 | Label threshold (novel vs adjacent) |
| `SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD` | `0.85` | 3 | Co-occurrence hard gate — 40M-scale recalibrated |
| `SYNTHESIS_MEMORY_SIMILARITY_THRESHOLD` | `0.85` | 4 | Internal duplicate threshold |
| `SYNTHESIS_COHERENCE_MODEL` | `sonnet-4.5` | 5 | LLM for coherence + skeptic |
| `SYNTHESIS_COHERENCE_MIN_LEVEL` | `MODERATE` | 5 | Minimum coherence gate |
| `SYNTHESIS_WEIGHT_COHERENCE` | `0.30` | 6 | Composite weight |
| `SYNTHESIS_WEIGHT_NOVELTY` | `0.40` | 6 | Composite weight |
| `SYNTHESIS_WEIGHT_DISTANCE` | `0.15` | 6 | Composite weight |
| `SYNTHESIS_WEIGHT_STRUCTURAL` | `0.15` | 6 | Composite weight |
| `SYNTHESIS_NOVELTY_W_CLAIM` | `0.25` | 6 | Novelty sub-weight |
| `SYNTHESIS_NOVELTY_W_COOCCURRENCE` | `0.30` | 6 | Novelty sub-weight |
| `SYNTHESIS_NOVELTY_W_SPECIFICITY` | `0.25` | 6 | Novelty sub-weight |
| `SYNTHESIS_NOVELTY_W_INTERNAL` | `0.20` | 6 | Novelty sub-weight |
| `SYNTHESIS_COMPOSITE_MIN_SCORE` | `0.65` | 6 | Minimum composite gate (raised from 0.40) |
| `SYNTHESIS_CONVERGENCE_STRONG_PATHS` | `3` | 7 | Paths for CONVERGING status |
| `SYNTHESIS_CONVERGENCE_STRONG_SOURCES` | `2` | 7 | Sources for CONVERGING status |
| `SYNTHESIS_LOG_ALL_REJECTIONS` | `True` | — | Log every rejection |
| `SYNTHESIS_DEFAULT_BATCH_SIZE` | `100` | — | Batch runner size |

### Generator (Cross-Store)

| Constant | Default | Purpose |
|----------|---------|---------|
| `SYNTHESIS_GENERATOR_ENABLED` | `True` | Master toggle |
| `SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION` | `5` | Target candidates per shutdown |
| `SYNTHESIS_GENERATOR_LLM_CONCURRENCY` | `5` | Max parallel bridge articulation calls |
| `SYNTHESIS_GENERATOR_MIN_GRAPH_NODES` | `20` | Graph sparsity guard |

### Retrieval Generator

| Constant | Default | Purpose |
|----------|---------|---------|
| `SYNTHESIS_RETRIEVAL_ENABLED` | `True` | Master toggle for retrieval-based synthesis |
| `SYNTHESIS_STRUCTURAL_QUERY_MAX_TOKENS` | `100` | Max tokens for structural query extraction LLM call |
| `SYNTHESIS_RETRIEVAL_K` | `5` | Number of FAISS results per structural query |
| `SYNTHESIS_RETRIEVAL_MIN_SIMILARITY` | `0.25` | Minimum cosine similarity for FAISS results |
| `SYNTHESIS_BRIDGE_ON_ACCEPT` | `True` | Create provisional graph edge on filter acceptance |
| `SYNTHESIS_BRIDGE_RELATION` | `"structural_parallel"` | Relation type for provisional bridge edges |

### Graph Walk Generator

| Constant | Default | Purpose |
|----------|---------|---------|
| `GRAPH_WALK_ENABLED` | `True` | Master toggle for graph walk synthesis |
| `GRAPH_WALK_MIN_BRIDGE_EDGES` | `40` | Minimum bridge edges to activate walk generator |
| `GRAPH_WALK_PERSONAL_RETURN_BIAS` | `2.0` | Node2Vec-style bias toward personal nodes |
| `GRAPH_WALK_HUB_DEGREE_THRESHOLD` | `15` | Degree above which hub dampening applies (log-scale penalty) |
| `GRAPH_WALK_MIN_DOMAINS` | `2` | Minimum distinct domain categories a walk must touch |
