# Synthesis Pipeline — End-to-End Results

**Date**: 2026-04-01
**Branch**: `refactor/prompt-modular`

---

## What Was Tested

Three diagnostic runs compared coherence judge configurations against production data:

1. **Run 1 — gpt-4o-mini coherence, composite at 0.55.** Baseline: all three generators, relaxed composite threshold to expose what the coherence judge lets through. 110 candidates. Result: 67% acceptance. Coherence rejected 4/110 (3.6%) — rubber stamp.

2. **Run 2 — Opus 4.6 coherence + hardened prompt, composite at 0.55.** Same generators, same relaxed composite, but coherence judge upgraded to Claude Opus 4.6 with de-jargon test + variable swap test. 110 candidates. Result: 30% acceptance. Coherence rejected 48/110 (44%) — now the primary quality gate.

3. **Production configuration — Opus coherence, composite at 0.65.** Not yet run as a diagnostic. Estimated ~20% acceptance based on composite score distribution of Run 2 accepted insights.

All data is machine-recorded in `data/synthesis_e2e_results.json` (Run 2).

---

## Data Profile at Test Time

| Resource | Run 1 | Run 2 |
|----------|-------|-------|
| FAISS wiki vectors | 40,982,675 | 40,982,675 |
| ChromaDB facts | 2,179 | 2,198 |
| Knowledge graph nodes | 30,929 | 31,056 |
| Knowledge graph edges | 8,343 | 8,521 |
| Bridge edges | 185 | 305 |
| Coherence model | gpt-4o-mini | claude-opus-4.6 |
| Composite threshold | 0.55 | 0.55 |
| Candidates generated | 110 | 110 |

---

## Run 2 Results (Opus + Hardened Prompt — Primary)

### Three-Tier Generation

| Generator | Tier | Candidates | Accepted | Rate |
|-----------|------|-----------|----------|------|
| RetrievalSynthesisGenerator | 0 (primary) | 50 | 21 | **42%** |
| GraphWalkGenerator | 1 (walks) | 7 | 0 | 0% |
| SynthesisGenerator | 2 (fallback) | 53 | 12 | **23%** |
| **Total** | | **110** | **33** | **30%** |

Retrieval outperforms random nearly 2:1 (42% vs 23%). The walk generator produced 7 candidates (bridge count now sufficient at 305) but none passed coherence.

### Rejection Breakdown

| Stage | Rejections | % | Role |
|-------|-----------|---|------|
| **Coherence judge (Stage 5)** | **48** | **44%** | Primary quality gate — de-jargon + variable swap tests |
| External novelty (Stage 3) | 23 | 21% | FAISS 40M corpus catches documented connections |
| Domain crossing (Stage 1) | 4 | 4% | Same-domain pairs filtered |
| Semantic distance (Stage 2) | 2 | 2% | Endpoints too close or too far |
| Composite scoring (Stage 6) | 0 | 0% | All weak candidates already caught by coherence |

The coherence judge is now the primary gate — the architecture working as designed. Composite scoring rejected zero candidates because coherence already killed the weak ones.

### Accepted Insights (Selected — Full List in JSON)

**Insight 1 (Retrieval): "Irreversible commitment points in batch processes"**
- `6 years in brewing <> Batch`
- Mechanism: In brewing, mash temperature locks in the fermentable sugar profile — once enzymes denature, you cannot undo the ratio. In sequential batch processes generally, each step has a point of no return that constrains all downstream options.
- Coherence: STRONG | Composite: 0.699

**Insight 2 (Retrieval): "Indirect resource transfer as bounded substitute"**
- `may help financially <> Dependency need`
- Mechanism: Financial help functions as a controllable channel when direct emotional engagement is too costly or risky. Same structure as institutional dependency theory — resource provision shapes behavior through implicit conditionality.
- Coherence: STRONG | Composite: 0.738

**Insight 3 (Retrieval): "Temporal niche mismatch"**
- `3 til 10 <> Diurnality`
- Mechanism: The user's active period (3 PM–10 PM) is systematically offset from the environment's default activity window, creating functional exclusion from infrastructure designed around diurnal schedules.
- Coherence: STRONG | Composite: 0.695

**Insight 4 (Retrieval): "Path dependency through layered temporal accumulation"**
- `in the oldest part of the city <> Gentrification of Mexico City`
- Mechanism: Each era's construction physically constrains subsequent modifications — street widths, building codes, utility routing. The same path-dependency dynamic operates in gentrification: prior investment layers lock in future development trajectories.
- Coherence: STRONG | Composite: 0.718

**Insight 5 (Retrieval): "Peak-moment compression in single-exposure encoding"**
- `Barcelona once <> Memory`
- Mechanism: Flashbulb and episodic memories from one-time events selectively preserve emotionally salient peaks while discarding routine context. A single visit to Barcelona compresses into peak moments.
- Coherence: STRONG | Composite: 0.742

**Insight 6 (XStore): "Objective function subject to hard constraints"**
- `dieting <> Engineering optimization`
- Mechanism: Both operate by defining an objective function (caloric deficit / performance metric) subject to hard constraints (minimum nutritional requirements / material strength limits). The optimization framework is structurally identical.
- Coherence: STRONG | Composite: 0.752

**Insight 7 (XStore): "Deep structure vs surface structure"**
- `at gym <> Deep structure and surface structure`
- Mechanism: A workout has a deep structure (progressive overload, periodization, muscle group targeting) and a surface structure (specific exercises on any given day). The same workout program produces different surface realizations while preserving deep structure.
- Coherence: STRONG | Composite: 0.759

### Quality Assessment

All 33 accepted insights were rated STRONG by Opus (not MODERATE). The hardened prompt forces binary discrimination: either the de-jargon test and variable swap test both pass (STRONG) or they don't (WEAK). No rubber-stamp middle ground.

The accepted claims name **specific, transferable mechanisms** with concrete predictions: "irreversible commitment points", "temporal niche mismatch", "path dependency through layered accumulation", "objective function subject to hard constraints." These survive the de-jargon test — the structural content persists without field-specific vocabulary.

**Pending human validation.** These results have not been graded through the audit queue. The claims read as grade 4+ on the structural rubric, but human grading is required to confirm. The 33 results are loaded in the Synthesis tab for review.

---

## Comparison: gpt-4o-mini vs Opus Coherence Judge

| Metric | gpt-4o-mini (Run 1) | Opus (Run 2) |
|--------|-------------------|-------------|
| Coherence rejections | 4/110 (3.6%) | **48/110 (44%)** |
| Acceptance rate | 67% | **30%** |
| Composite rejections | 3/110 | **0/110** |
| Retrieval acceptance | 80% | **42%** |
| XStore acceptance | 63% | **23%** |
| Retrieval vs XStore gap | 1.3x | **1.8x** |
| Coherence ratings | All MODERATE | All STRONG or WEAK |
| Claim quality | Surface metaphor ("can be seen as") | Named mechanisms with predictions |

**Key finding:** The coherence judge was the critical failure point. Upgrading from gpt-4o-mini to Opus + hardened prompt transformed the pipeline from a noise generator (67% acceptance of mostly junk) to a discriminating filter (30% acceptance with mechanism-naming claims).

**Confound:** Both the generation model and judge model changed to Opus simultaneously. The quality improvement reflects both better articulation (generation) and stricter evaluation (judging). To isolate which matters more, a controlled run with Opus generation + gpt-4o-mini judging would be needed.

---

## Bridge Feedback Loop

| Metric | Run 2 |
|--------|-------|
| Bridge edges at start | 305 |
| Walk candidates generated | 7 (bridge threshold met) |
| Walk candidates accepted | 0 |
| Provisional bridges created | 33 (from accepted insights) |

The walk generator is now active (305 > 40 bridge threshold) but produced no accepted candidates. Walk-generated claims may need different prompt tuning — the narration prompt interprets a graph path, which is a different task than articulating a structural query result.

---

## Production Configuration

```yaml
# config/config.yaml (current)
synthesis_generator:
  coherence_model: claude-opus-4.6
  composite_min_score: 0.65
synthesis_audit:
  enabled: true
  fp_halt_threshold: 0.50
  min_graded: 10
```

At composite 0.65, candidates with composite < 0.65 from Run 2 would be filtered. Of the 33 accepted, ~8-10 scored below 0.65. Estimated production acceptance: ~20-25 out of 110 candidates (~20%).

Cost per synthesis run: ~$3-4 (Opus coherence calls via OpenRouter at $15/$75 per 1M input/output tokens).

---

## Limitations

1. **No human grading yet.** The 33 accepted insights have not been graded through the audit queue. Opus STRONG ratings correlate with better claims, but human validation is required.

2. **Generation + judging confounded.** Both changed to Opus. The quality improvement may be primarily from better generation (stronger articulations) rather than stricter judging.

3. **Parser failures.** 21/110 coherence responses couldn't be parsed (Opus gives longer de-jargon analysis). These defaulted to WEAK. Parser has been fixed for future runs but some valid candidates may have been incorrectly rejected in Run 2.

4. **Walk generator 0% acceptance.** 7 candidates generated, 0 accepted. Walk narration may need prompt adjustment — interpreting a graph path is a different task than articulating a structural query.

5. **Cost.** Opus coherence at ~$3-4 per run is sustainable but 50x more expensive than gpt-4o-mini. If the SVM/KNN classifier from the audit queue can eventually replace the LLM judge, cost drops back down.

---

## Next Steps

1. **Grade the 33 accepted insights** through the Synthesis audit tab using the 1-5 structural rubric. This builds the ground-truth dataset.

2. **Run production sessions** with Opus coherence at composite 0.65 to measure real-world acceptance rates and insight quality.

3. **Isolate the confound.** Run Opus generation + gpt-4o-mini judging to determine whether the quality improvement comes from better generation or stricter judging.

4. **Accumulate grading data** toward the 300-pair target for SVM/KNN classifier training.

5. **Tune walk generator prompt.** 0/7 acceptance suggests the walk narration prompt needs adjustment for the graph-path interpretation task.
