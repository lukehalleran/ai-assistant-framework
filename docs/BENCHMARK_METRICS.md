# Retrieval Benchmark Metrics

Tracks Daemon's retrieval quality and latency over time. Updated by running `python scripts/benchmark_retrieval.py`. Full history in `data/benchmark_history.json`.

All quality metrics computed over the 13 retrieval cases only (6 intent-only cases excluded from recall/precision/MRR). Intent accuracy computed over all 19.

## Current Metrics (2026-05-13)

### Quality (n=13 retrieval cases)

| Metric | Value |
|--------|-------|
| Recall@1 | **0.58** |
| Recall@3 | **1.00** |
| Recall@10 | **1.00** |
| Precision@10 | **0.11** |
| MRR | **0.79** |
| Intent accuracy | **100%** (19/19) |
| Cases passed | **19/19** |

### Rank Distribution (n=14 expected items)

| Bucket | Count | Pct |
|--------|------:|----:|
| Rank 1 (top) | 8 | 57% |
| Rank 2-3 | 6 | 43% |
| Rank 4-5 | 0 | 0% |
| Rank 6-10 | 0 | 0% |
| Rank 11+ | 0 | 0% |

100% of correct memories in top 3.

### Latency

| Component | p50 | p90 | p95 |
|-----------|-----|-----|-----|
| Retrieval + scoring | 23.9ms | 26.7ms | 27.7ms |
| ChromaDB lookup | 8.6ms | 9.2ms | 9.3ms |
| Prompt build | 767ms | 1,480ms | 1,601ms |

### Quality by Intent

| Intent | R@1 | R@3 | R@10 | MRR | Pass |
|--------|-----|-----|------|-----|------|
| temporal_recall | 0.75 | 1.00 | 1.00 | 1.000 | 2/2 |
| factual_recall | 0.67 | 1.00 | 1.00 | 0.778 | 3/3 |
| project_work | 0.50 | 1.00 | 1.00 | 0.750 | 2/2 |
| creative_exploration | 0.50 | 1.00 | 1.00 | 0.750 | 2/2 |
| emotional_support | 0.50 | 1.00 | 1.00 | 0.750 | 2/2 |
| technical_help | 0.50 | 1.00 | 1.00 | 0.750 | 2/2 |
| casual_social | -- | -- | -- | -- | 2/2 |
| meta_conversational | -- | -- | -- | -- | 2/2 |
| general | -- | -- | -- | -- | 2/2 |

`--` = intent-only test (no retrieval requirement).

### Per-Case Detail

| Case | R@1 | R@3 | MRR | Ranks |
|------|-----|-----|-----|-------|
| factual_dogs_name | 1.00 | 1.00 | 1.000 | [1] |
| factual_language_preference | 1.00 | 1.00 | 1.000 | [1] |
| temporal_recent_today | 0.50 | 1.00 | 1.000 | [2, 1] |
| temporal_last_week | 1.00 | 1.00 | 1.000 | [1] |
| emotional_stressed_at_work | 1.00 | 1.00 | 1.000 | [1] |
| technical_segfault_debug | 1.00 | 1.00 | 1.000 | [1] |
| creative_dashboard_ideas | 1.00 | 1.00 | 1.000 | [1] |
| project_auth_status | 1.00 | 1.00 | 1.000 | [1] |
| emotional_anxious_cant_sleep | 0.00 | 1.00 | 0.500 | [2] |
| technical_python_async | 0.00 | 1.00 | 0.500 | [2] |
| creative_new_features | 0.00 | 1.00 | 0.500 | [2] |
| project_api_update | 0.00 | 1.00 | 0.500 | [2] |
| factual_sister_birthday | 0.00 | 1.00 | 0.333 | [3] |

## Historical Metrics

### Quality Over Time

| Date | R@1 | R@3 | R@10 | MRR | Note |
|------|-----|-----|------|-----|------|
| 2026-05-12 | -- | -- | 1.00 | 0.52 | Baseline (R@1/R@3 not yet tracked) |
| 2026-05-13 (temporal fix) | 0.50 | 0.69 | 1.00 | 0.66 | Two-regime temporal decay + metric bug fixes |
| 2026-05-13 (scoring tuning) | **0.58** | **1.00** | **1.00** | **0.79** | Stemming + tag bonus + weight rebalance |

### Rank Distribution Over Time

| Date | Rank 1 | Rank 2-3 | Rank 4-5 | Rank 6-10 | Rank 11+ |
|------|--------|----------|----------|-----------|----------|
| 2026-05-13 (temporal fix) | 7 (50%) | 3 (21%) | 2 (14%) | 2 (14%) | 0 (0%) |
| 2026-05-13 (scoring tuning) | **8 (57%)** | **6 (43%)** | **0 (0%)** | **0 (0%)** | **0 (0%)** |

### Tuning Set Progress

| Case | Rank (baseline) | Rank (tuned) | MRR (baseline) | MRR (tuned) |
|------|-----------------|--------------|----------------|-------------|
| factual_sister_birthday | 7 | **3** | 0.143 | **0.333** |
| emotional_anxious_cant_sleep | 8 | **2** | 0.125 | **0.500** |
| technical_python_async | 4 | **2** | 0.250 | **0.500** |
| creative_new_features | 5 | **2** | 0.200 | **0.500** |
| project_api_update | 3 | **2** | 0.333 | **0.500** |

### What Changed (2026-05-13)

**Temporal decay fix** (memory_scorer.py):
- Scorer now uses `time_manager.current()` instead of `datetime.now()` for age calculation
- Two-regime decay: small anchors (<=48h) use flat plateau, large anchors (>48h) peak at anchor age

**Stemming** (memory_scorer.py):
- `_stem()` suffix stripper so "anxious"/"anxiety", "deployed"/"deployment" match in token overlap
- Applied to both query tokens and memory tokens

**Tag-keyword bonus** (memory_scorer.py):
- If stemmed query tokens match stemmed memory tags, adds up to +0.15 to continuity score

**Weight rebalancing** (intent_classifier.py):
- Shifted weight from recency → continuity across 5 intents to stop recent irrelevant memories outranking older relevant ones

## How to Run

```bash
python scripts/benchmark_retrieval.py
```

Results saved to:
- `data/benchmark_results.json` — full latest run with per-case detail
- `data/benchmark_history.json` — append-only summary history

## Test Corpus

30 seed memories + 19 test cases in `tests/fixtures/retrieval_benchmarks.yaml`:
- 13 retrieval cases (have `must_retrieve` assertions) across 6 intent types
- 6 intent-only cases (casual_social, meta_conversational, general) — test classification only
