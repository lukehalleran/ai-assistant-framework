# Synthesis Pipeline — End-to-End Results

**Date**: 2026-04-01
**Branch**: `refactor/prompt-modular`

---

## What Was Tested

The three-tier synthesis pipeline was run against production data to measure acceptance rates, rejection distribution, and insight quality across generators.

**Single experiment:** RetrievalSynthesisGenerator (Tier 0) + GraphWalkGenerator (Tier 1) + SynthesisGenerator (Tier 2) running in parallel, with the calibrated coherence judge (requires MODERATE = names a real mechanism concretely applied to both domains). 3 runs of 10 candidates each = 30 total.

All data is machine-recorded in `data/synthesis_e2e_results.json`.

---

## Data Profile at Test Time

| Resource | Value |
|----------|-------|
| FAISS wiki vectors | 40,982,675 |
| ChromaDB facts | 2,179 |
| Knowledge graph nodes | 30,929 (29,705 wikidata + 1,209 personal + 15 wiki-retrieved) |
| Knowledge graph edges | 8,343 |
| Bridge edges (personal ↔ wikidata) | 6 (after cleanup of 129 garbage bridges) |
| LLM | Production model (via ModelManager) |
| Candidates generated | 30 (15 Tier 0 + 0 Tier 1 + 15 Tier 2) |

---

## Results

### Three-Tier Generation

| Generator | Tier | Candidates | Accepted | Rate |
|-----------|------|-----------|----------|------|
| RetrievalSynthesisGenerator | 0 (primary) | 15 | 2 | 13% |
| GraphWalkGenerator | 1 (walks) | 0 (inactive) | 0 | — |
| SynthesisGenerator | 2 (fallback) | 15 | 2 | 13% |
| **Total** | | **30** | **4** | **13%** |

Both generators achieved identical 13% acceptance rates. The retrieval generator did not outperform the random generator quantitatively in this run. See "Qualitative Difference" below for where they diverge.

GraphWalkGenerator produced zero candidates because only 6 bridge edges exist (minimum threshold: 40). After cleanup of 129 garbage bridges (from low-confidence WikidataEntityMapper matches), the walk generator is gated until the bridge feedback loop accumulates sufficient quality bridges.

### Filter Outcomes

| Metric | Value |
|--------|-------|
| Candidates in | 30 |
| Accepted | 4 |
| Rejected | 26 |
| **Acceptance rate** | **13%** |

### Rejection Breakdown by Stage

| Stage | Rejections | Role |
|-------|-----------|------|
| Text sanity (Stage 0) | 0 | Generator output is well-formed |
| Domain crossing (Stage 1) | 0 | All pairs were cross-domain |
| Semantic distance (Stage 2) | 0 | All pairs within [0.20, 0.90] range |
| External novelty (Stage 3) | 12 | FAISS 40M corpus caught known/documented connections |
| Internal novelty (Stage 4) | 0 | No prior synthesis results to collide with |
| Coherence judge (Stage 5) | 1 | LLM judged claim as structurally weak |
| Composite scoring (Stage 6) | 13 | Multi-signal composite below 0.65 threshold |

The primary rejection gates are **composite scoring** (43%) and **novelty** (40%). Coherence rejected only 1 candidate (3%), meaning most candidates that survive novelty produce claims the LLM rates as at least MODERATE. This holds for both generators — the calibrated coherence judge is not the bottleneck at this graph density.

### Accepted Insights

**Insight 1 (Tier 0 — Retrieval): "Historical layering and cultural shifts"**
- Personal fact domain: urban geography / living situation
- Wikipedia domain: heritage studies
- Mechanism: Historical layering — how successive waves of cultural change leave stratified traces in both urban neighborhoods and heritage preservation policy. The same dynamic of incremental overwriting applies to how cities evolve and how cultural heritage is selectively preserved or erased.
- Coherence level: MODERATE
- Composite: 0.651

**Insight 2 (Tier 0 — Retrieval): "Conditional dependency"**
- Personal fact domain: family / financial relationships
- Wikipedia domain: welfare economics / dependency theory
- Mechanism: Conditional dependency — the structural parallel between family financial support with implicit conditions and welfare systems that create dependency traps. Both involve resource provision that shapes behavior through implicit rather than explicit conditionality.
- Coherence level: MODERATE
- Composite: 0.655

**Insight 3 (Tier 2 — XStore): Pattern recognition as personal narrative**
- Personal fact domain: daily events ("earlier today")
- Wikipedia domain: pattern (mathematics)
- Claim: Events or actions "earlier today" as a pattern of behavior contributing to personal narrative, like how mathematical patterns form through consistent rules.
- Coherence level: MODERATE
- Composite: 0.657

**Insight 4 (Tier 2 — XStore): Emotional historicization**
- Personal fact domain: cognitive/emotional sensations
- Wikipedia domain: historicization
- Claim: Emotional experiences gradually transformed into a narrative that informs future perceptions — a personal historicization process.
- Coherence level: MODERATE
- Composite: 0.673

### Qualitative Difference Between Generators

The retrieval generator's accepted insights name **specific, falsifiable mechanisms** ("conditional dependency", "historical layering") that can be independently verified — a domain expert can assess whether the named mechanism genuinely applies to both domains.

The XSTORE generator's accepted insights pass the MODERATE coherence bar but rely on **broader structural claims** ("pattern of behavior" ↔ "mathematical patterns", "emotional narrative" ↔ "historicization"). These are defensible connections but less precisely mechanistic.

This qualitative gap is the retrieval generator's real contribution: structural query extraction forces the LLM to identify a mechanism **before** searching, and the adversarial evaluation tests whether the mechanism actually applies. The random generator sometimes produces MODERATE-quality claims, but the mechanism naming is less specific.

**Important caveat:** Both generators achieved 13% acceptance in this run (n=15 each). A qualitative advantage with no quantitative advantage means the retrieval generator's value is in **insight depth**, not acceptance rate. This may diverge with more data — 15 candidates per generator is too small for statistical confidence on rate differences.

---

## Prior Calibration Baseline

The closest documented baseline for the SynthesisGenerator alone comes from the co-occurrence gate recalibration run (`SYNTHESIS_CALIBRATION_PLAN.md`, 2026-03-31):

| Metric | Calibration baseline | Current E2E |
|--------|---------------------|-------------|
| Generator | SynthesisGenerator only | 3-tier (Retrieval + Walk + XStore) |
| Coherence judge | Pre-MODERATE (WEAK accepted) | MODERATE required |
| Co-occurrence threshold | 0.85 (just recalibrated) | 0.85 |
| Candidates | 45 (15 × 3 runs) | 30 (10 × 3 runs) |
| Acceptance rate | 22% (10/45) | 13% (4/30) |
| Novelty rejections | 11 | 12 |
| Coherence rejections | 20 | 1 |
| Composite rejections | 4 | 13 |

**Key differences:**
- The calibration baseline used the **pre-MODERATE** coherence judge, which accepted WEAK claims. 20/34 candidates that survived novelty were still rejected by the weaker coherence standard.
- Tightening to MODERATE would reject more of the candidates that previously passed coherence, but how many more is unknown — the calibration run wasn't re-run with the MODERATE judge against the same data.
- The shift from coherence-dominated rejection (calibration: 20/45) to composite-dominated rejection (E2E: 13/30) could reflect either the stricter coherence judge filtering out weak candidates earlier (so the ones that survive are stronger overall) or simply different random samples.

**There is no machine-recorded baseline for the SynthesisGenerator alone under the MODERATE coherence judge.** The calibration plan's 22% result used the old coherence standard. Drawing conclusions about how much the coherence recalibration changed the random generator's acceptance rate requires a controlled re-run that hasn't been done.

---

## Bridge Feedback Loop

Accepted synthesis insights create provisional bridge edges in the knowledge graph:

| Metric | Before run | After run |
|--------|-----------|-----------|
| Total edges | 8,339 | 8,343 |
| Bridge edges (personal ↔ wikidata) | ~129 (pre-cleanup) → 6 (post-cleanup) | 6 + 4 provisional |

The 4 accepted insights each created a provisional bridge edge (weight=0.5). These edges mature to full weight (1.0) when independently rediscovered via convergence. As provisional bridges accumulate, the GraphWalkGenerator (Tier 1) will eventually have enough bridges (>=40) to activate, enabling walk-based synthesis that crosses the personal-wikidata boundary.

Growth path: Tier 0 (retrieval) seeds bridges → Tier 1 (walks) exploits bridges → more walk candidates → more bridges. At 4 provisional bridges per synthesis run, this requires ~9 more successful runs to activate the walk generator.

---

## Validation Against Calibration Plan Criteria

The calibration plan (`docs/SYNTHESIS_CALIBRATION_PLAN.md`, Phase 3, Step 3.1) defines these targets:

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| 1 | Candidates generated | >= 10 | 30 | **PASS** |
| 2 | Candidates with LLM articulation | >= 6 | 30 (all) | **PASS** |
| 3 | Stage 0-2 rejections (malformed) | <= 2 | 0 | **PASS** |
| 4 | Stage 3 rejections (novelty) | 2-5 | 12 | **ABOVE TARGET** |
| 5 | Stage 5 rejections (coherence) | 1-3 | 1 | **PASS** |
| 6 | Final acceptances | 1-4 | 4 | **PASS** |
| 7 | Acceptance rate | 10-30% | 13% | **PASS** |

Stage 3 rejections (12) exceeded the 2-5 target. With 40M FAISS vectors, the novelty gate catches more documented connections than the original target anticipated. This is correct behavior — the target was set before the full FAISS index was available.

---

## What This Shows

### 1. The filter pipeline works

All three primary gates are active and doing different work:
- **Novelty** (12 rejections) catches candidates whose connections are already documented in Wikipedia
- **Composite** (13 rejections) catches candidates with weak multi-signal profiles
- **Coherence** (1 rejection) catches candidates where the LLM articulated a structurally weak claim

The 13% acceptance rate falls within the 10-30% target band.

### 2. Retrieval-based synthesis produces qualitatively deeper output

Retrieval insights name specific mechanisms ("conditional dependency", "historical layering") that are independently verifiable. XSTORE insights pass the same coherence bar but with less mechanistic precision. The structural query extraction step forces mechanism identification before search, which produces more specific claims.

### 3. Both generators pass calibrated coherence at comparable rates

The SynthesisGenerator (random pairing) achieved the same 13% acceptance rate as the RetrievalSynthesisGenerator under the MODERATE coherence standard. Random pairing CAN produce MODERATE-quality claims — it does so less consistently in mechanistic specificity, but the coherence judge does not categorically reject random-pairing output.

### 4. The bridge feedback loop is operational

4 provisional bridge edges were created from accepted insights. The mechanism for graph growth through synthesis acceptance is working. The walk generator remains gated at 6 < 40 bridges but will activate as bridges accumulate.

### 5. Composite scoring is the primary bottleneck

13/30 candidates (43%) failed composite scoring — more than any other gate. Some of these may be interesting candidates with strong coherence but marginal novelty or distance scores. The 0.65 composite threshold may need adjustment as more data accumulates.

---

## Limitations

1. **Small sample size.** 15 candidates per generator in a single session. The 13% rate for both generators could diverge significantly with larger samples. Statistical confidence requires 50-75+ candidates per generator across multiple runs.

2. **No controlled baseline under MODERATE coherence.** The prior calibration baseline (22%, SynthesisGenerator only) used the pre-MODERATE coherence judge. We don't have a direct comparison of the random generator before and after coherence recalibration with all other parameters held constant.

3. **Composite threshold is tight.** At 0.65, composite scoring is the largest rejection gate. Some rejected candidates may have strong coherence but marginal novelty scores, especially from the retrieval generator (whose claims use Wikipedia-adjacent language that can trigger false novelty matches).

4. **Walk generator inactive.** Only 6 quality bridges exist. The walk generator requires 40+. At 4 provisional bridges per successful synthesis run, activation requires ~9 more runs. The bridge feedback loop is slow but operational.

5. **Novelty gate may over-reject retrieval candidates.** The retrieval generator searches Wikipedia by structural pattern, so its claims naturally use Wikipedia-adjacent language. High claim similarity scores may trigger false novelty rejections for connections that are structurally novel but phrased similarly to existing articles.

6. **Single LLM configuration.** All coherence judging used one model (gpt-4o-mini via openrouter). Different models may produce different coherence ratings, especially near the WEAK/MODERATE boundary.

---

## Next Steps

1. **Run the SynthesisGenerator alone under MODERATE coherence** (45 candidates, 3 runs) to establish a proper controlled baseline. This is the missing comparison that would quantify whether the retrieval generator has a real rate advantage or only a quality advantage.

2. **Increase sample size** to 50+ candidates per generator across multiple sessions to get statistical confidence on rate differences.

3. **Investigate composite threshold sensitivity.** Run the accepted candidates at 0.60 and 0.70 to understand how many interesting candidates are marginal.

4. **Track per-generator stage breakdowns.** The current rejection breakdown is aggregated across all generators. Per-generator breakdowns would reveal whether the two generators fail at different stages (e.g., XSTORE failing more at novelty, retrieval failing more at composite).
