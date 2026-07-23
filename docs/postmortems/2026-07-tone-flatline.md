# Postmortem: tone-flatline / distress amplification (2026-07)

Status: fixed. Stub — captured, not polished.

## Symptom
During a real distress session (medication-withdrawal, ~8 turns of short
messages), Daemon hardened transient feelings into a life-narrative and ended
almost every turn with an "excavating" question that dug further into the pain.
Owner flagged it as "mirroring or amplifying too much." The session opened with
the user saying they felt "a little better" and, eight turns later, arrived at
"[everything] fundamentally missing."

## The de-escalation curve finding
The damning signal: support level moved *inversely* to distress. Turn 1
("...but I am anxious", 12 words) correctly detected CONCERN/LIGHT SUPPORT. Every
*shorter*, sadder message after it (7 words, 3 words, 1 word) dropped back to
CONVERSATIONAL. Support engagement was inversely correlated with distress,
mediated almost entirely by **message word count** — the system got quieter and
more clinical exactly as the user sank.

## Root causes (two, independent, both dead-wires)
1. **Tone fast-path bypass** (`utils/tone_detector.py`). The `<8-word` latency
   optimization (commit 33246a2) returned CONVERSATIONAL unconditionally and
   ignored conversation history. A distress session built from terse turns
   therefore flatlined. Compounded by a dead escalation boost in the semantic
   path (`recent_distress = 0.5` gated on `> 0.5` — never fired).
2. **CrisisLevel→ToneLevel enum mismatch** (`core/context_pipeline.py`). The
   pipeline mapped tone via `ToneLevel.from_string(crisis_level.value)`, but
   `.value` is `"light_support"/"elevated_support"/"crisis_support"` while
   `from_string` only knew the `"HIGH"/"MEDIUM"/"CONCERN"` name-scale. **Every**
   CrisisLevel therefore mapped to CONVERSATIONAL, so the `EscalationTracker`
   was fed CONVERSATIONAL on every turn and `GROUNDING_PRESENCE`/
   `QUIET_COMPANIONSHIP` (the built-in cure for the excavation pattern) could
   never fire in production. This was found by the golden-transcript replay
   *after* the fast-path fix, because unit tests called `tracker.update(
   ToneLevel.CRISIS)` directly and bypassed the conversion.

Same class as prior incidents (`.intent_type` None, `semantic_score` 0): a
signal silently defaulting, hidden by tests that didn't exercise the real seam.

## Fix summary
- Tone stickiness: fast path now exits early only for recognizably-casual
  messages; `previous_tone` (threaded through the pipeline) + recent heavy-topic
  history keep distress sticky; non-casual terse replies mid-distress floor at
  CONCERN; explicit acks still decay. Dead escalation boost repaired.
- Escalation reachability: CONCERN-inclusive `consecutive_distress_count`
  (threshold 5) upgrades slow spirals to GROUNDING without touching
  `consecutive_elevated_count`; GROUNDING/QUIET copy forbids excavating
  questions. `ToneLevel.from_string` now accepts both encodings.
- Valence-aware retrieval (`memory/valence.py`): caps mood-congruent negative
  recall in `[RELEVANT MEMORIES]` during distress, backfills lower-affect hits.
- Runtime canary (`core/safety_canary.py`): log-only `SAFETY_CANARY_TONE_FLATLINE`
  when N consecutive negative-affect messages read as CONVERSATIONAL.

## Tests added
- `tests/integration/test_golden_distress_replay.py` — acceptance replay of the
  frozen transcript (`tests/fixtures/golden_distress_session.json`); asserts the
  intended trajectory; verified RED when either root cause is reintroduced.
- `tests/unit/test_tonelevel_from_string.py` — pins BOTH enum encodings.
- `tests/unit/test_anti_amplification.py` — stickiness, escalation, valence.
- `tests/unit/test_length_invariance.py` — length does not collapse distress.
- `tests/unit/test_safety_canary.py` — canary behavior.
- `tests/conftest.py` — safety-skip guard (no silent skips on safety tests).
- Un-skipped the 3 previously-dead semantic tone tests (now env-gated).

## Follow-ups
- **Length-invariance is only partially closed.** The strict "short within one
  level of long AND never below long" does NOT hold universally, even in-session:
  verbose distress legitimately reaches HIGH while terse forms sit at MEDIUM, and
  the keyword harm-route caps some short forms at CONCERN and skips the semantic
  boost. Closing it needs threshold/routing calibration — deliberately deferred
  (no threshold tuning from tests). Revisit with real-usage data.
- **Threshold calibration from real use**: `distress_threshold=5`,
  `valence_retrieval.max_negative_fraction=0.5`, `canary.consecutive_threshold=4`
  are first guesses, not calibrated.
- **Valence lexicon upgrade**: if canary/displacement logs show clinical-vocab
  false positives (e.g. medication/therapy terms scored negative), replace the
  lexicon scorer with something better than substring matching.
- **The keyword route bypasses the escalation boost** — a short keyword-matched
  distress message ("i feel hopeless") caps at CONCERN in-session. Consider
  letting session distress lift keyword-routed CONCERN too.
