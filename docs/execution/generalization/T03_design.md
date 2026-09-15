# T03 — negated crisis phrase under news framing: CONCERN for that turn only (design decision)

## Status

- **Owner decision, 2026-09-13**, on T02's open decision point (quoted): "a negated crisis phrase with news framing gets CONCERN (light support) for that turn only. It must not re-arm the sticky distress floor, so the next turn starts fresh. Scope that exclusion to this new trigger only; every other CONCERN trigger keeps its current carry-over. Add a probe showing that a neutral follow-up turn after such a message stays conversational."
- **Author:** parent design, generalization lane.
- **Scheduling:** runs immediately after T02.
- **Landing constraint:** T01, T02 and T03 land together.
  - With T02 but without T03, the negated-news CONCERN would re-arm the floor.
  - With T01 but without T02, negated and third-party phrases over-escalate.

## Trigger split (implemented in T02)

| Trigger | When (Stage 0 fired) | This turn | Carry-over |
|---|---|---|---|
| `observational_negated_crisis` | The only first-person evidence is a negated HIGH crisis phrase | CONCERN | **none** (this batch) |
| `observational_first_person_distress` | A qualifying non-negated mild first-person hit is present, with or without a negated HIGH hit | CONCERN | normal, like every other CONCERN trigger |

## Verified carry-over paths (candidate tree)

1. **Previous tone.**
   - `core/context_pipeline.py` `ContextPipeline._detect_tone` stores every turn's level in `self._last_tone_level`, updates `_floor_chain` (incremented only for `distress_sticky_floor`, reset otherwise), and persists via `_persist_tone(level, trigger)`.
   - The next turn passes `previous_tone`, and `utils/tone_detector.py` `_session_in_distress` treats any CONCERN as session distress, which lets the sticky floor fire.
2. **Restart seeding.** `_load_persisted_tone` skips only trigger `distress_sticky_floor`.
3. **Heavy history.**
   - `_recent_distress_from_history` → `_heavy_row_is_distress_evidence` re-arms session distress from any fresh `is_heavy_topic` row whose text carries a first-person marker.
   - "people are suffering but I don't want to die" carries "I".
   - Its heavy flag does not come from the heuristic. The parent ran the real `utils/query_checker.py` `_is_heavy_topic_heuristic` on it: `False`, no heavy-keyword hits.
   - It can still be flagged in live use. `core/context_pipeline.py:925-936` calls `analyze_query_async(query, self.model_manager)`, which consults `_classify_heavy_topic_llm` whenever the heuristic says not heavy (`utils/query_checker.py:1086-1095`).
   - So this path is live whenever that classifier flags the turn. The read-time re-check below does not depend on which classifier set the flag.
   - The pipeline's own comment records that withholding `previous_tone` alone is not enough: the history path re-fires the floor (2026-08-28).
4. **Distress counter.**
   - `core/escalation_tracker.py` `EscalationTracker.update` increments `consecutive_distress_count` for every CONCERN turn except trigger `distress_sticky_floor`, which holds it.
   - `core/orchestrator.py` passes `tone_trigger` into the tracker.

## Decision: the negated-news turn is transparent to carry-over ("hold")

- **ContextPipeline `_detect_tone`.** When this turn's trigger is `observational_negated_crisis`:
  - return this turn's CONCERN;
  - do NOT update `_last_tone_level`;
  - do NOT change `_floor_chain`;
  - do NOT call `_persist_tone`.

  `_load_persisted_tone` also skips the trigger, as a defensive guard for state written by any other path.
- **Heavy history (read-time re-check).**
  - `_heavy_row_is_distress_evidence` re-classifies the row's own text with T02's deterministic classifier. A row that classifies as `observational_negated_crisis` is not distress evidence.
  - No storage schema change. This follows the same read-time doctrine as the 2026-09-05 first-person check and the 2026-09-08 pasted-code check.
- **EscalationTracker.** `observational_negated_crisis` holds `consecutive_distress_count` without incrementing it, exactly like `distress_sticky_floor`.
- **One exported constant.** The trigger value lives in `utils/tone_detector.py`; consumers import it rather than repeating the string.
- **Unchanged:**
  - all `distress_sticky_floor` handling;
  - every other CONCERN trigger, including `observational_first_person_distress`;
  - the gap reset, the floor chain budget, and `allow_sticky_floor`.

### Hold, not reset (owner-confirmed 2026-09-13)

The owner confirmed this reading, quoted: "a negated crisis phrase with news framing does not arm carry-over, and it does not clear earlier genuine distress either. Earlier CONCERN carry-over stays in effect; no full reset."

- **No earlier distress:** the next turn starts fresh (probe rows 1–3).
- **Genuine distress on an earlier turn:** that carry-over survives the negated-news turn in between (probe hold row).

## Probe (parent, 2026-09-13)

- **Files:** script `docs/execution/generalization/probes/t02_followup_probe.py`; output `docs/execution/generalization/probes/t02_followup_probe_output.txt`.
- **Real code exercised:** `utils/tone_detector.py` at the R05 final digest `98adfca7…`; `detect_crisis_level(model_manager=None)` with a locally cached embedder (offline, no provider); `_recent_distress_from_history` with the real `_heavy_row_is_distress_evidence`; the real `EscalationTracker`.
- **Simulated:**
  - T02's tier classification, re-implemented with the deployed matchers;
  - the designed row re-check, a wrapper around the real function;
  - the designed pipeline hold, by passing the `previous_tone` the designed `_detect_tone` would pass (None when no earlier distress).
- **Heavy flag:** history rows are marked `is_heavy_topic=True`, the worst case.

| Turn 1 (Stage 0 fires) | Neutral follow-up | Carry via previous tone (today's rule) | Carry via heavy row (today's rule) | Designed |
|---|---|---|---|---|
| people are suffering but I don't want to die → `observational_negated_crisis` | the weather was grey this morning | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | **CONVERSATIONAL / semantic** |
| (same) | I watched a documentary about whales last night | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | **CONVERSATIONAL / semantic** |
| (same) | my sister is visiting next week | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | **CONVERSATIONAL / semantic** |
| people are suffering and I feel hopeless → `observational_first_person_distress` (control) | the weather was grey this morning | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor (carry-over kept) |
| (same) | I watched a documentary about whales last night | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor |
| (same) | my sister is visiting next week | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor | CONCERN / distress_sticky_floor |

- **Hold row:** genuine CONCERN at turn N-1 ("…I feel hopeless"), negated news at turn N (held), neutral at turn N+1 → CONCERN / distress_sticky_floor. The N-1 carry-over is preserved.
- **Tracker today** (`consecutive_distress_count` after one CONCERN turn):
  - organic `harm_score…` → 1;
  - `distress_sticky_floor` → 0 (hold);
  - `observational_negated_crisis` → 1. It must become 0 (hold).

### Implemented-code probe (parent, T03 review)

After integration the owner's requested probe was re-run through the REAL implemented code: `ContextPipeline._detect_tone` for two consecutive turns, `model_manager=None`, tone state in a temp directory.

- Files: `docs/execution/generalization/probes/t03_pipeline_followup_probe.py`, with output in `…_output.txt`.
- Candidate digests: `core/context_pipeline.py` `853ff419…`, `core/escalation_tracker.py` `8d612937…`, `utils/tone_detector.py` `3d65d1a4…`.

Results:
- After "people are suffering but I don't want to die" (CONCERN, `observational_negated_crisis`), all six neutral follow-ups resolve **CONVERSATIONAL / semantic**. That is three follow-ups, each run with and without a heavy history row.
- After "people are suffering and I feel hopeless" (CONCERN, `observational_first_person_distress`), all six resolve CONCERN / `distress_sticky_floor`.
- Hold row (genuine CONCERN → negated news → neutral): CONCERN / `distress_sticky_floor`.
- Tracker: organic mild CONCERN then negated news → `consecutive_distress_count` 1, held.

## Acceptance tests (T03 batch; clean and wrapped user text)

- **Real two-turn pipeline test.** Drive `ContextPipeline._detect_tone`, following the seams in `tests/unit/test_tone_stickiness_reset.py` and `tests/unit/test_tone_floor_self_latch.py`, with `model_manager=None` and a sandboxed tone state.
  - Turn 1 "people are suffering but I don't want to die" → CONCERN with trigger `observational_negated_crisis`; `_last_tone_level` unchanged; nothing persisted.
  - Turn 2, each of the three neutral follow-ups → CONVERSATIONAL.
  - Repeat with turn 1's heavy history row present → CONVERSATIONAL.
- **Control.** Turn 1 "people are suffering and I feel hopeless" → turn 2 neutral → CONCERN / `distress_sticky_floor`.
- **Hold.** Organic CONCERN at N-1, negated news at N, neutral at N+1 → the floor applies.
- **Restart.** Persisted tone state after a negated-news turn equals the prior state, and the loader skips the trigger if it is present.
- **Tracker.**
  - `observational_negated_crisis` holds the count: not incremented, not reset.
  - `observational_first_person_distress` increments it.
  - The existing `distress_sticky_floor` tests stay unchanged.
- **Row re-check.** A negated-news row → False; a mild-distress row → True; a legacy row with no text field → True (unchanged).
- **Existing tests stay green unmodified:** `test_tone_floor_self_latch.py`, `test_tone_stickiness_reset.py`, `test_escalation_deescalation_fix.py`, `test_anti_amplification.py`, and the T01/T02 tone tests.

## BC-28 / BC-58

- **BC-28.** The negated-news signal can no longer feed any carry-over path: previous tone, restart seed, heavy history or distress counter. Its trigger remains visible to DM-10.
- **BC-58 sibling sites the batch must check:**
  - every `distress_sticky_floor` comparison: `core/context_pipeline.py` (chain and loader), `core/escalation_tracker.py`, and `core/orchestrator.py` (hand-off only);
  - every reader of `_session_in_distress` and `_recent_distress_from_history`.
- **Out of scope, reported for the owner:** other readers of the stored heavy flag — `memory/thread_manager.py` (session opener) and `core/prompt/section_instructions.py` (heavy-topic prompt). They are not tone carry-over.

## Batch

| Batch | Files | ~Lines |
|---|---|---|
| T03 | `utils/tone_detector.py` (row re-check); `core/context_pipeline.py` (`_detect_tone` hold, loader skip); `core/escalation_tracker.py` (hold); new `tests/unit/test_tone_negated_news_no_carryover.py` | ~200 |
