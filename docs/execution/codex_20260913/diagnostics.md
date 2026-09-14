# Batch C — offline telemetry diagnostics

`scripts/diagnostic_rollup.py` reads one caller-supplied JSONL file and writes a fixed-schema aggregate to stdout. It uses only the Python standard library and imports no Daemon modules. It resolves and checks both the supplied path and symlink target for any path component named `data` before opening, requires a regular file, and never chooses an input or output path itself. It bounds each line to 1 MiB and stops after 100,000 rows by default. A reached cap is explicit in the result.

The rollup accepts an explicit local date and IANA timezone. It reports malformed JSON, oversized rows, test rows, invalid timestamps, out-of-window rows, and legacy naive timestamps separately. Aware timestamps are converted to the selected zone; naive timestamps are interpreted in that zone and counted. Missing or invalid measurements have null summaries or coverage gaps; they are not converted to zero. Numeric durations reject booleans, negative values, non-numbers, NaN, and infinities. Median and p90 use linear interpolation, matching `scripts/latency_rollup.py`. Phase and task durations are summarized independently and never added together as wall time.

Output labels come from fixed mode, model, phase, and task allowlists. Unknown models/modes collapse to `other`; unknown timing keys are counted and dropped to bound retained samples. The output contains no input text, session identifiers, arbitrary keys, exception details, or free-text reasons. Slow samples contain only source row numbers, validated wall durations, and allowlisted mode/model labels.

## A07 producer handoff

The current producer (`utils/turn_telemetry.py`, owned by A07) should emit a versioned, stable receipt with these fields:

- `turn_id`: opaque per-turn correlation ID, stable across logs for that turn and unrelated to session/user identity.
- `build_sha` and `telemetry_schema_version`: exact build and producer schema identifiers.
- `outcome_status`: typed enum such as `success`, `partial`, `failed`, `not_run`, or `cancelled`; `outcome_reason_code`: bounded enum/category, never exception prose.
- `stage_timings_s`: named monotonic elapsed durations for prepare, retrieval, provider first token (`ttft_s`), provider completion, verification, and total wall. Wall is measured from a monotonic clock; nested stages are not summed.
- `model_id`, `input_tokens`, `output_tokens`, `cache_tokens`, `cost_usd`, and `provider_retries`, with unknown values omitted/null and zero reserved for observed zero.
- Any routing or verification receipt needed to explain a decision should use typed decision/source/reason-code fields; user text, prompts, retrieved passages, responses, and raw exception bodies must not enter telemetry.

Acceptance checks for A07: (1) a single turn's correlation ID joins debug and turn telemetry while differing across turns; (2) success, partial provider refusal, provider failure, and not-run produce distinct typed outcomes, including when one parallel provider call succeeds and another fails; (3) absent token, cache, cost, TTFT, and retry receipts remain unknown, while measured zero remains zero; (4) all durations are nonnegative monotonic elapsed values and nested stages may exceed neither their real interval nor be blindly summed into total wall; (5) malformed typed fields are rejected or counted invalid without becoming empty-success; (6) positive canaries show free text is absent and negative canaries detect any attempted query, response, prompt, retrieved text, or exception-message emission; (7) rollup coverage demonstrates the new fields and continues to bucket unknown labels; (8) no reserved producer file is changed by this batch.

This batch is an offline reader and report only. It does not close BC-20, BC-47, BC-69, BC-70, or BC-72: producer receipts, operational alerting, and owner-integrated CI coverage remain follow-up work. The observed source lacked consistent turn IDs, build versions, token/cost accounting, and complete typed outcome receipts; absent optional receipts are measurement gaps, not evidence of a runtime failure.

## Validation record

Focused unittest command requested by the batch owner:

```text
systemd-run --user --scope -p MemoryMax=512M /home/lukeh/.pyenv/versions/3.11.8/bin/python -m unittest discover -s tests/unit -p test_diagnostic_rollup.py
```

The test run is parent-coordinated because only one capped test process may run at a time. The dated aggregate is retained in `diagnostic_summary.json`; its counts are limited to the selected date and timezone and do not reproduce source rows.

Parent validation: 10 focused tests passed under the 512 MB scope. The capped
aggregate read 2,762 rows, excluded 736 test rows and 1,937 other-date rows,
and selected 89 production turns. There were no malformed or oversized rows
and no truncation. Of 87 measured wall times, median was 21.062 seconds,
linearly interpolated p90 42.4536 seconds, maximum 138.667 seconds.
