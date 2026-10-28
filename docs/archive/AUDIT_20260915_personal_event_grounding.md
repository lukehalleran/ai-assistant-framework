# Personal-event confabulation — 2026-09-15

## State and conclusion

Audit and repair design only; runtime behavior is **not fixed**. Read
`DEVELOPMENT_WORKFLOW.md`, `BUG_CLASSES.md`, and both generalization audits.
Replayed the supplied reply through deployed guard functions without model
calls or store mutations. The observed turn is the enhanced response recorded
at `2026-09-15T16:46:46-05:00`.

The reply promoted a contemplated task and the assistant's own recommendation
into a completed user event. The prompt already explicitly prohibited this.
Existing guards cover public factual errors, assistant tool actions, and a
narrow attachment-recency case; they do not enforce evidence for general
claims about what the user did. Classes: BC-46, BC-48, BC-58; BC-70 for the
misleading receipt. Persistence creates a BC-75 feedback risk; no subsequent
re-ingestion or repetition was established in this audit.

## Evidence in the supplied prompt

| Reply claim | Actual evidence | Assessment |
|---|---|---|
| Résumé reworked enough to be done | User: “I could just upload what I have to get something up, but it needs rework.” Assistant recommended uploading as-is. | Completion invented; the user's stated assessment was replaced with the assistant's assessment. |
| Uploaded it | Assistant: “then the one task is upload-as-is.” No user completion report. | Recommendation/possibility promoted to completion. |
| Had the tripwire conversation | Recent conversation contains that exchange. | This part is supported by the supplied history. |
| Nothing left today that cannot wait | No complete current task/deadline inventory. | Absence of evidence promoted to certainty about the rest of the day. |
| Sleep deprivation explains the experience, using 6.5 hours/3am | Those figures occur in a prior assistant reply in the visible recent context. | The excerpt does not independently establish the figures or the causal explanation; it does not prove the figures false either. |

Likely generation behavior: the model filled out an encouraging accomplishment
recap, borrowing task nouns and the recommendation from prior conversation.
That is an interpretation of the text, not access to the model's internal
reasoning. The guard gaps below are reproducible code facts. The timings do
not establish retrieval latency or a model-provider bug as the cause.

## Verified control flow

1. `core/response_guidance.py:22` already says earlier advice is history, not
   evidence. `core/grounding_check.py:51` already prohibits claiming planned
   tasks completed. Both appear in the supplied system prompt. Adding the same
   instruction again is not a demonstrated closure (BC-46).
2. `core/response_planner.py:175` skips CONCERN, which selects LIGHT SUPPORT.
   The live log records CONCERN and telemetry has `response_plan: null`.
   `gui/handlers.py:4750` requires a plan for review; that review is itself
   log-only. There is no evidence that a planner invented these completions.
3. `core/grounding_check.py:165` admits truth-stance language, selected
   scientific causal claims, years/percentages, or endorsement of those
   shapes. The exact reply returns **False**.
4. Even if admitted, `gui/handlers.py:3389` passes the current message, reply,
   runtime clock, and optional tool material. The enhanced caller at
   `gui/handlers.py:4901` supplies no conversation evidence. Furthermore,
   `_build_verifier_prompt` at `core/grounding_check.py:231` asks about
   established falsehoods, explicitly abstaining on unverifiable claims.
   Missing support for a personal event is a different question.
5. Live telemetry records `grounding_mode: log_only`, matching
   `config/config.yaml:210`. Switching to `correct` alone would fix neither
   the filter nor the evidence/contract gap. BC-50 documents why automatic
   corrections were disabled: previous false corrections.
6. `gui/handlers.py:3281` turns a still-pending background check into
   `complete`, including when the prefilter returned early. This turn has
   `grounding_status: complete` with no prefilter/verifier-fired fields.
   Complete here is task completion, not successful factual verification.
7. The guard siblings also miss: `detect_completion_claims` checks assistant
   tool claims; `claims_fresh_upload` checks a narrower user attachment shape.
   Its handler uses Daemon's active-document registry, which cannot establish
   whether a résumé was uploaded to an external career site. Expanding the
   word matcher alone would use the wrong evidence source.
8. Storage completed according to the live log. The shared conversation
   annotator leaves this reply unmarked, so retrieval can later expose it as
   an ordinary assistant reply. That is a feedback risk, not proof the
   completion was extracted into the user profile.

## Read-only reproduction

These are diagnostic outcomes, **not passing behavioral regression tests**.
Both the exact response and a wrapped/indented variant produced:

```json
{"has_checkable_claims": false, "completion_claims": 0,
 "claims_fresh_upload": false, "annotator_changes_reply": false}
```

Exact query:

> No idea how it's almost 5, feel like I've thunk a total of like 3 thoughts today

Exact response (retained for the eventual outcome fixture):

> That's the sleep-deprivation time warp — 6.5 hours on a 3am crash doesn't feel like a day, it feels like a blur with occasional horizontal breaks. And honestly, you did think more than 3 thoughts: you reworked the resume enough to call it done, uploaded it, and had a whole tripwire-framework conversation. Fried brains just don't log those as"thinking."
>
> Low bar for the rest of the evening. Nothing left today that can't survive until tomorrow.

Functions called: `core.grounding_check.has_checkable_claims`,
`core.action_claim_guard.detect_completion_claims`, `claims_fresh_upload`,
and `annotate_unverified_action_claim`. No external LLM was invoked.
The deployed `gui.handlers._apply_grounding_check` also returned `(None, "")`
with empty telemetry in both `log_only` and `correct` modes, for both forms;
a model stub that would raise on any call was never invoked.

## General repair design

**Proposed, not implemented:** a shared personal-claim support boundary,
independent of tone and response planning (CM-01), using source-backed event
state and bounded verification (CM-03), initially measured in shadow mode
(CM-04). This is a provenance/state problem, not a résumé vocabulary miss.

1. Preserve the original role, timestamp, source ID, and exact supporting
   span from structured conversation records. Keep assistant suggestions,
   user intentions, reported completions, and tool receipts distinguishable.
   Do not create a trusted event ledger solely by summarizing assistant text.
2. Assess a draft's personal claims against those records. A candidate event
   needs an actor, action, object, time/episode, and modality/status. Use a
   bounded semantic extraction/check step for open-ended action language;
   validate returned source IDs/spans mechanically. A matching substring
   alone does not prove entailment. Unknown support remains **unknown**, not
   “the user did not do it.” Later direct corrections take precedence.
3. Distinguish support/contradiction/insufficient evidence. Check user
   completions against user reports, assistant executions against matching
   execution receipts, and external uploads against the external task's
   evidence. An attachment in Daemon establishes only an attachment there.
4. Unsupported completion claims should be omitted or qualified without
   inventing their negation. Universal claims about free time need a complete
   obligation inventory; ordinarily omit them. Preserve supported adjacent
   content. No automatic substantive correction until precision is measured
   against counterexamples (BC-50).
5. Run the same boundary for enhanced and agentic answers, including
   emotional replies. Enforcement requires checking before delivery and
   persistence; a background log-only check cannot retract streamed text.
   Reuse the existing buffering infrastructure where appropriate. On checker
   failure, emit an explicit unavailable receipt; decide the delivery policy
   separately and never describe fail-open text as verified.
6. Carry provenance/support status into subsequent retrieval and generated
   summaries so repetition cannot become independent corroboration. A
   legacy assistant statement without evidence remains an assistant claim.
7. Record distinct outcomes for skipped/no candidate, checked, unavailable,
   and failed. Emit candidate/support counts, evidence IDs, latency and
   delivery action; redact any text. Budget measurements must include this
   added model work. No extra call for every retrieved memory.

### Acceptance cases and sibling inventory

The implementation must turn the exact excerpt above into a regression
fixture with its user/assistant context, timestamps, and clean plus
wrapped/indented forms. Assert the final **delivered and stored claims**, not
just classifier routing or a prompt sentence. Include:

- “could upload”, “will upload”, assistant “upload it”, and user “uploaded it”;
- quoted completion, negation, conditional plans, cancellation, partial work;
- same action on a different object or date, ambiguous pronouns, recurring
  tasks, and a later correction;
- real completion preserved, supported discussion preserved, and no invented
  “not done” claim when evidence is missing;
- unrelated domains (sent a message, paid a bill, attended an appointment,
  finished a repair), novel action wording, and non-completion emotional text;
- external upload versus Daemon attachment; incomplete deadline inventories;
- verifier timeout/invalid result, both delivery modes, both answer routes,
  and a subsequent retrieval of the previously unsupported response.

BC-58 sibling search was run across `core/`, `gui/`, `utils/`, `knowledge/`
and `memory/`. Relevant sites:

- `gui/handlers.py`: enhanced/agentic delivery, action guard, background
  grounding, final storage and telemetry.
- `core/prompt/gatherer_memory.py`, `formatter.py`,
  `gatherer_knowledge.py`: conversation, merged-content and self-note
  provenance annotation, including planner/decision consumers.
- `utils/daily_notes_generator.py` and `memory/memory_consolidator.py`:
  derived summaries/narratives. Existing `status_claims` handles only selected
  profile conflicts; `completed_plan_claims` handles the **opposite** error
  (a completed plan still narrated as pending), not invented completion.

## Implementation (2026-09-15, same day)

The shared boundary proposed above shipped: `core/personal_claim_check.py`
(evidence + auditor + omission), `utils/personal_claim_provenance.py`
(receipt whitelist + read-time marker at every conversation consumer), and
the handler wiring on both routes with the mode captured at ingress. Default
is `enabled: true, mode: log_only` — receipts only; `correct` is opt-in.
The F6 receipt gap is closed: a grounding check that never ran records
`skipped` with a reason, never `complete`. Tests:
`tests/unit/test_personal_claim_{check,provenance,delivery}.py`. Still open:
precision measurement on a labeled set, a live canary before `correct`, and
a post-restart live probe. Full entry: `CLAUDE_CHANGELOG.md` (2026-09-15).

## Resume note / remaining work (as written before the implementation)

Diagnosis and read-only guard replay complete. No production guard, config,
store, or model setting changed. Next work is implementation of the shared
evidence contract plus outcome fixtures and precision evaluation above.
Do not call this closed after adding a prompt sentence or another upload
regex. A live post-restart probe remains necessary after an eventual fix.

Validation: `python scripts/check_bug_classes.py scan --root .` reports OK
(no new/stale gated findings); `git diff --check` is clean. This is a docs-only
batch, so no behavioral test suite is claimed. The required session snapshot
captured git state but its whole-workspace hashing was interrupted; it did not
produce a manifest. Explicit pre-edit copies of all four existing edited docs
are in `/tmp/sep15-grounding-doc-preimages/`. The session audit confirmed
unchanged branch/HEAD and skipped the absent manifest comparison. The two
pre-existing untracked docs were left untouched.
