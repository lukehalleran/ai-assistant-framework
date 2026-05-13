# Retrieval Benchmark Metrics

Tracks Daemon's retrieval quality and latency over time. Updated by running `python scripts/benchmark_retrieval.py`. Full history in `data/benchmark_history.json`.

All quality metrics computed over retrieval cases only (intent-only cases excluded from recall/precision/MRR). Intent accuracy computed over all cases.

## Current Metrics (2026-05-13)

### Quality (n=72 retrieval cases, 93 seed memories)

| Metric | Value |
|--------|-------|
| Recall@1 | **0.39** |
| Recall@3 | **0.65** |
| Recall@10 | **1.00** |
| MRR | **0.58** |
| Intent accuracy | **100%** (96/96) |
| Cases passed | **96/96** |

### Rank Distribution (n=86 expected items)

| Bucket | Count | Pct |
|--------|------:|----:|
| Rank 1 (top) | 33 | 38% |
| Rank 2-3 | 28 | 33% |
| Rank 4-5 | 11 | 13% |
| Rank 6-10 | 14 | 16% |
| Rank 11+ | 0 | 0% |

71% of correct memories in top 3. 100% in top 10.

### Latency

| Component | p50 | p90 | p95 |
|-----------|-----|-----|-----|
| Retrieval + scoring | 25.5ms | 28.9ms | 31.1ms |
| ChromaDB lookup | 8.6ms | 9.3ms | 9.5ms |
| Prompt build | 891ms | 1,907ms | 2,165ms |

### Quality by Intent

| Intent | R@1 | R@3 | R@10 | MRR | Cases |
|--------|-----|-----|------|-----|-------|
| project_work | 0.50 | 0.80 | 1.00 | 0.692 | 10 |
| technical_help | 0.40 | 0.90 | 1.00 | 0.658 | 10 |
| emotional_support | 0.45 | 0.65 | 1.00 | 0.634 | 10 |
| factual_recall | 0.47 | 0.68 | 1.00 | 0.609 | 19 |
| temporal_recall | 0.44 | 0.50 | 1.00 | 0.532 | 8 |
| meta_conversational | 0.14 | 0.36 | 1.00 | 0.454 | 7 |
| creative_exploration | 0.12 | 0.50 | 1.00 | 0.362 | 8 |
| casual_social | -- | -- | -- | -- | 8 |
| general | -- | -- | -- | -- | 8 |

**Weakest areas:** creative_exploration (MRR=0.36, R@1=0.12) and meta_conversational (MRR=0.45). Creative queries use vague language that embeds close to many memories. Meta queries route through a special corpus path that the benchmark doesn't fully exercise.

## Historical Metrics

### Quality Over Time

| Date | Corpus | R@1 | R@3 | R@10 | MRR | Note |
|------|--------|-----|-----|------|-----|------|
| 2026-05-12 | n=13 | -- | -- | 1.00 | 0.52 | Baseline |
| 2026-05-13 | n=13 | 0.58 | 1.00 | 1.00 | 0.79 | Temporal fix + scoring tuning |
| 2026-05-13 | **n=72** | **0.39** | **0.65** | **1.00** | **0.58** | Expanded corpus with hard negatives |

### How Misleading Was n=13?

| Metric | n=13 | n=72 | Overstatement |
|--------|------|------|---------------|
| Recall@1 | 0.58 | 0.39 | +49% |
| Recall@3 | 1.00 | 0.65 | +54% |
| MRR | 0.79 | 0.58 | +36% |

The n=13 corpus had zero hard negatives and mostly single-item retrieval. It tested "can the scorer find a needle in 30 memories" — not "can it pick the right needle from 5 similar needles in 93 memories."

## Test Corpus

93 seed memories + 96 test cases in `tests/fixtures/retrieval_benchmarks.yaml`:
- 72 retrieval cases across 7 intent types (8-19 per intent)
- 24 intent-only cases (casual_social, meta_conversational, general)
- Hard negatives: family disambiguation, pet disambiguation, Python subtopic discrimination, project component discrimination, emotional state discrimination
- Multi-item retrieval: queries expecting 2-3 memories
- Cross-collection: queries targeting summaries or conversations

## How to Run

```bash
python scripts/benchmark_retrieval.py
```

Results saved to:
- `data/benchmark_results.json` — full latest run with per-case detail
- `data/benchmark_history.json` — append-only summary history
