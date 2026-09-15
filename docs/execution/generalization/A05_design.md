# A05 atomic correction delivery — design decision (F04 / G12 acceptance 3–6)

Status: parent design decision (generalization lane), based on a read-only survey on 2026-09-13 of the uncommitted candidate on base `328a8ec`. This is a design, not an implementation. It runs after A04 (strict verifier JSON), which owns `core/grounding_check.py` until A04 is integrated.

## Verified gaps

1. **Flagged draft already on screen.** In `GROUNDING_MODE="correct"` the draft streams as cumulative `message` events:
   - enhanced route: `gui/handlers.py:4207,4231,4243`;
   - agentic route: `gui/handlers.py:3659`.
   Review runs only after the stream (`:4570` enhanced, `:3914` agentic). The final chunk then replaces the bubble with the revised or suffixed text (`:4619`, `:3959`).
2. **Suffix fallback.** When the integrator returns None (timeout, error, length ratio, "⚠️" in the rewrite, dropped date, weekday mismatch, input over 4000 chars; `core/grounding_check.py:899-985`), the suffix from `build_grounding_correction` (`grounding_check.py:838-848`) is appended to the draft (`handlers.py:3263-3266,4575-4576`).
3. **Cancel hole.** A client disconnect cancels `__anext__` (`api/chat_service.py:166-172`). `CancelledError` escapes `except Exception` (`handlers.py:4577,4622,3267`; `verify_grounding` re-raises at `grounding_check.py:764-767`). The enhanced `finally` then stores the unreviewed `final_output` (`handlers.py:4679-4720`) without checking `debug_emitted`.
4. **Display, stored and indexed text diverge.** Display-only links and cards (`:4541,4601,3801`) plus three storage sanitizers apply. The enhanced route has no parity guard; the agentic route has a log-only one (`:3981`).
5. **Verifier failure (parent-verified).** `_apply_grounding_check_for_delivery` returns `_no_action` when the verdict is None (`handlers.py:~3203`). Text is unchanged, and `verify_grounding` records `grounding_status="failed"` / `grounding_failure_reason`. No suffix and no marker.

## Decision: design A — buffer in correct mode

- Read the mode once per turn into `ctx` at turn start (no mid-turn flip).
- **correct mode:** both route loops convert answer-content chunks into progress-only chunks ("Checking facts…") and hold the latest content. The final text is emitted ONCE after review. If the route ends without a debug/final chunk (the error/empty early returns at `:4138,4272,4279,4329,3630,3670`), flush the held content so errors still show.
- **log_only (default):** streaming unchanged.
- **Scope:** the change lives in `gui/handlers.py` only. No SSE schema, web client, or store-writer change. Gradio `/admin` inherits it, because buffering happens inside `handle_submit`.
- **Rejected: design B (provisional stream + replacement event).** The user can read or copy a flawed draft (BC-45 across time), and it needs a new client contract (SSE schema, React reducer, live regions, Gradio has no equivalent).

### Contract points

- **Flagged + integrator succeeds:** deliver the integrated revision once.
- **Flagged + integrator fails/times out/disabled:** ONE integrated fallback, never draft + suffix:
  - (a) splice the correction in place when the claim sentence is located uniquely in the draft;
  - (b) otherwise a single standalone corrective reply that does not contain the flawed claim.
  - Receipt `grounding_status` records `fallback` and the reason.
- **Not flagged, or suppressed verdict:** deliver the draft once.
- **Verifier failure (verdict None):** deliver the draft once, unmodified; receipt `grounding_status="failed"` (existing). No suffix or marker (BC-47 visibility lives in receipts and debug, not in appended text).
- **Cancel before review completes (correct mode only):** store NO assistant text for the turn (a draft never enters retrieval). Record a receipt `delivery="cancelled_before_review"`. log_only keeps today's storage behavior.
- **Atomicity:** one canonical `final_text` per turn. `display_text == stored_text == indexed_text == final_text` holds after one declared normalization (the existing display-only-decoration strip, `handlers.py:2739`, plus the storage sanitizers). Tests compare normalized equality. The enhanced route gains the parity guard the agentic route has.
- **No verdict or draft in retrieval:** verdict text stays in telemetry and debug only (already true, redacted to 300 chars).
- **Unchanged:** `GROUNDING_MODE` default `log_only`; telemetry keys other than the two receipt values above.

## Batches (disjoint files; no behaviorally active partial migration)

| Batch | Scope | Files | ~Lines |
|---|---|---|---|
| A05a | Integrated-fallback builder (splice-or-standalone), INACTIVE (no caller) | `core/grounding_check.py`, new `tests/unit/test_grounding_integrated_fallback.py` | ~200 |
| A05b | Atomic switch: mode on `ctx`, buffer wrapper for both routes, suffix branches replaced by the A05a builder, cancel storage gate, enhanced parity guard | `gui/handlers.py`; `tests/unit/test_grounding_wiring.py`, `test_grounding_log_only.py`, plus parent-approved amendments `test_sep09_speed_batch.py`, `test_audit0831_fixes.py` (they pin the suffix) | ~420 |
| A05c | Delivery atomicity tests only: both routes × integrate on/off × integrator fail/timeout × verifier failure × cancel during review × log_only. Assert no answer `message` before review in correct mode, and `complete` == history == stored == corpus/Chroma fakes == debug record (normalized) | new `tests/unit/test_grounding_delivery_atomicity.py` | ~350 |
| A05d | Retire `build_grounding_correction` and its tests. The catalog update (BC-45/BC-50) is the class-guard owner's: record a note in the packet; do NOT edit `docs/BUG_CLASSES.md` | `core/grounding_check.py`, `tests/unit/test_grounding_check.py` | ~80 |

**Shared seam:** `gui/handlers.py` (A05b). No other batch may own it concurrently. A07 and the F-series batches that touch handlers are scheduled after A05.

**Probes:** Q13 depends on A04; Q14 (forced integrator timeout) and Q15 depend on A05.

## Revision 2026-09-13 (parent): split A05b into three sequential batches

Why: A05b's ~420-line estimate sits at the 450 hard stop. A05a, which was estimated at ~200 lines, came in at 588. `gui/handlers.py` is the riskiest shared seam in the lane. Each step below leaves correct mode no worse than today, so the split creates no harmful intermediate state; log_only, the default, is unchanged throughout.

| Batch | Scope | Why it is safe alone |
|---|---|---|
| A05b-1 | Fallback switch. In `_apply_grounding_check`, a flagged verdict whose integrator is disabled or returns None goes through `build_integrated_fallback` (spliced or standalone), and the text is delivered through the existing `revised` path. Receipt: `grounding_status="fallback"` plus `kind:reason`. The two suffix-append call sites are removed. Pinned tests are amended. | Today the correct-mode final chunk already replaces the streamed bubble with revised or suffixed text; afterwards it replaces it with the integrated fallback, never draft plus suffix. |
| A05b-2 | Buffered delivery in correct mode. The mode is read once into `ctx`; both route loops emit progress-only chunks and hold content; held content is flushed on the error and empty early returns. | Removes the visible flawed draft. Cancel-time storage is still the pre-existing gap, closed next. |
| A05b-3 | Cancel storage gate (`delivery="cancelled_before_review"`, no assistant text stored in correct mode) plus the enhanced-route parity guard. | Closes gaps 3 and 4. |

A05c (the atomicity matrix) and A05d (retiring `build_grounding_correction`) follow unchanged. The shared-seam rule still applies: no other batch owns `gui/handlers.py` until A05b-3 is integrated.

A05b-2 parent review finding F1 is added to A05b-3's scope as its first failing-test item:
- **Defect.** `_buffer_grounding_draft` flushes the last held chunk when a wrapped route ends without a final chunk. That flush is correct for the enhanced route, and for the agentic early returns that set `ctx.handled`. It is wrong when `_run_agentic_search` fails and returns unhandled: the dispatcher then falls through to the enhanced route, so a partial, unreviewed agentic draft would be flushed onto the wire first.
- **Fix.** Do not flush when the agentic route ends unhandled.
- **Why deferred.** A05b-2 is at 437 of 450 lines. The interim is still no worse than before A05b-2, which streamed the whole draft.

Queue note: T01/T02 (`utils/tone_detector.py`) run between A05b-2 and A05b-3. They do not touch `gui/handlers.py`.

Standalone wording: A05a's standalone lead must not imply the user saw a prior answer (A05a review finding F5). That wording is exact once A05b-2 buffers the draft. Between A05b-1 and A05b-2 it replaces a streamed draft; this is acceptable because correct mode is not the default.
