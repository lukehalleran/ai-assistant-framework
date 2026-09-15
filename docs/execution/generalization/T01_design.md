# T01 — first-person crisis language outranks observational framing (design decision)

Status:
- **Scheduling:** owner decision 2026-09-13 to run after A05b-2 and before A05b-3.
- **Source:** escalation E-R05-1 (`docs/execution/generalization/batches/R05.md`, parent section; `class_guard_responses/CGR-20260913-003-2.md`, finding 1).
- **Scope:** this design is by the generalization parent. It touches `utils/tone_detector.py` and tests only; the `gui/handlers.py` seam is not touched.

## Verified defect (BC-28 incident chain: a tone level that is too low starves the downstream safety systems)

`detect_crisis_level` runs Stage 0 (`_check_observational_language`) before the Stage 1 keyword check. Stage 0 fires on ≥2 observational markers, or on ≥2 third-person pronouns with no first-person marker. When it fires, the turn returns CONVERSATIONAL and Stage 1 never runs.

The parent probe used the deployed functions on the candidate tree, with no embedder:

| Message | Stage 0 | Stage 1 alone | Delivered today |
|---|---|---|---|
| "people are suffering and I want to die" | fires | MEDIUM | CONVERSATIONAL |
| "According to the news, officials say millions are struggling, and honestly I can't go on" | fires | MEDIUM | CONVERSATIONAL |
| "People are dying in the news and I don't want to be here anymore" | fires | MEDIUM | CONVERSATIONAL |
| "the news is awful, people are suffering, and I'm ready to die" | fires | MEDIUM | CONVERSATIONAL |

`core/prompt/builder.py` consults `_check_keyword_crisis` separately, but only to keep full prompt context. The tone level that selects the support mode remains CONVERSATIONAL.

## Why a plain reorder is rejected

`HIGH_CRISIS_KEYWORDS` mixes first-person ideation phrases ("want to die", "kill myself", "end my life") with topical nouns ("suicide", "suicidal"). Running Stage 1 first would escalate news discussion:

- "According to officials, suicide rates rose among young people"
- "The news article says people are suicidal after the policies"
- "According to the news, millions of people want to die in that war"

That kind of over-escalation is the upstream false positive BC-28 warns re-arms the distress floor.

## Decision

Stage 0 does NOT short-circuit when the message contains a **first-person HIGH crisis hit**; detection continues to Stage 1 unchanged. Otherwise Stage 0 behaves exactly as today. A HIGH hit (from `_HIGH_MATCHER.iter_hits`) is first-person when either:

- (a) **The phrase itself is first-person.** The matched keyword contains a first-person token ("kill myself", "end my life", "better without me", "wish i was dead").
- (b) **A first-person subject sits just before it.** A first-person token appears within `FIRST_PERSON_CRISIS_WINDOW_TOKENS` (3) tokens before the hit, inside the same sentence (stopping at `.`, `!`, `?` or a newline).
  - Matches: "I want to die", "and I really just want to die", "I'm ready to die".
  - Does not match: "I think the news coverage of suicide…", where the subject is 5 tokens away.

Constraints:

- **No new vocabulary (BC-76).** First-person detection reuses the module's existing word-bounded `_HISTORY_FIRST_PERSON_RE`. The window is a named numeric constant with a comment explaining why it is conservative.
- **Scope.**
  - HIGH vocabulary only.
  - MEDIUM/CONCERN vocabulary under observational framing is out of scope and recorded as a sibling for the owner.
  - `_check_observational_language` itself is unchanged, so its helper tests keep their meaning.
- **No self-teaching from mixed messages (BC-29).**
  - When the override is what let a message reach Stage 1, Stage 1 returns its level but does NOT call `_learn_tone_exemplar`.
  - A message mixing news framing with first-person crisis language is not a clean exemplar; teaching it could make future news discussion score as crisis.
  - Messages that reach Stage 1 without the override keep today's learning.
- **Receipts.**
  - An overridden turn's `ToneAnalysis.trigger` stays the Stage 1 trigger, so no new telemetry vocabulary is needed.
  - The explanation may say that observational framing was overridden by first-person crisis language, without echoing the message text.
- **Unchanged.**
  - The Stage 1 scoring route and thresholds.
  - The negation-blind harm scan (R05 parent correction).
  - The semantic, LLM and floor stages.
  - `core/prompt/builder.py`.
- **Docstrings.** Update the module overview's Stage 0 description (BC-71).

## Known, accepted limitations

- **Negated first-person crisis phrases** under observational framing now reach Stage 1: "people are suffering but I don't want to die" → MEDIUM. This matches how the same sentence scores without observational markers today (harm scan is negation-blind by design) and errs toward support.
- **A first-person reader near a topical noun** can match rule (b) ("I read suicide statistics") when the subject is within the window. "I read an article that says suicide rates are rising" is already MEDIUM today, because Stage 0 does not fire on it, so T01 does not change it.
- **MEDIUM/CONCERN first-person distress** under observational framing ("people are suffering and I feel hopeless") stays CONVERSATIONAL. It is recorded for the owner.
- **A possessive or plural first-person token** also qualifies under rule (b), because the reused `_HISTORY_FIRST_PERSON_RE` includes "me", "my", "mine", "we", "our" and "ours". Example: "the news says my students want to die". This errs toward support. It is kept, rather than adding a narrower pronoun list (BC-76).

## Acceptance tests (deployed `detect_crisis_level(model_manager=None)`, plus the new helper; clean and wrapped forms)

- **Escalate** (level ≠ CONVERSATIONAL and trigger ≠ `observational_language`): the four table rows above, plus "According to the article people are struggling and I really just want to die". Include rows where rule (a) alone applies and rows where rule (b) alone applies.
- **Stay CONVERSATIONAL with trigger `observational_language`:**
  - "According to officials, suicide rates rose among young people"
  - "The news article says people are suicidal after the policies"
  - "I think the news coverage of suicide is irresponsible, people are struggling"
  - "They say people are suffering and they can't go on"
  - "According to the news, millions of people want to die in that war"
  - "People are struggling with the new policies, according to officials"
- **Sentence scoping:** a first-person token in a previous sentence does not qualify a later third-person hit. Example: "Wait, is that me? Civilians, society want to die." Stage 0 fires on the markers "civilians" and "society".
  - Correction (parent, T01 review): the example originally given here, "I read it. They say people want to die there.", is not observational. `_check_observational_language` returns False with no markers, so the message never reaches Stage 0 and resolves to MEDIUM through Stage 1 on both base and candidate. The T01 worker substituted the valid example above.
- **BC-29:** an overridden turn does not call `_learn_tone_exemplar` (spy or fake); a non-observational first-person crisis turn still does.
- **Existing tests stay green unmodified:** `_check_observational_language` helper tests, R05's `test_tone_keyword_boundaries.py`, and anti-amplification.

## Follow-up (owner request 2026-09-13)

- Both sides of the limitations above are addressed by batch **T02**, which runs immediately after T01: `docs/execution/generalization/T02_design.md`.
- T01 itself is unchanged.
- **Landing constraint:** T01 and T02 land together. On its own, T01 raises negated and third-party crisis phrases under news framing to MEDIUM, and T02 corrects that.

## Batch

| Batch | Files | ~Lines |
|---|---|---|
| T01 | `utils/tone_detector.py` (Stage 0 gate in `detect_crisis_level`, one new private helper, Stage 1 learning guard, module docstring); new `tests/unit/test_tone_first_person_crisis_override.py` | ~150 |
