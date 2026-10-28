# September 6 conversation audit — working record

## Workflow kept in context

Canonical workflow: `docs/DEVELOPMENT_WORKFLOW.md`. Trace the code that ran;
fix root causes with tests calling deployed functions; run targeted suites,
ruff and shared-contract guards; probe the deployed functions; update doctrine,
changelog and handoff. Owner commits, applies store changes and restarts.
No store mutation or historical-data deletion is part of this audit.

## Accepted direction

Evidence should support a conversational answer. Detailed evidence reports
remain available when requested. Avoid another full-answer rewrite call when
the final synthesis can consume the conversation and evidence directly.

Medication discussion must weigh both benefits and harms and distinguish drug
classes. Planned stimulant breaks are a legitimate option to evaluate, not a
categorically forbidden topic. The failure is unsupported personal endorsement
and invented mechanisms, not the existence of that option. NICE NG87 1.10.1–3
supports individualized review and trial discontinuation; its rationale notes
possible symptom worsening and reduced adverse effects. This is not evidence
for a particular monthly schedule for this individual.

Sources checked September 6, 2026:
- https://www.nice.org.uk/guidance/ng87/chapter/recommendations#review-of-medication-and-discontinuation
- https://www.nice.org.uk/guidance/ng87/chapter/Rationale-and-impact#review-of-medication-and-discontinuation

## Verified roots and planned repairs

1. `core/agentic/gate.py`: both LLM pattern branches accept a classifier flag
   without checking whether the user requested analysis. Preserve explicit
   history requests; reject an unsolicited report for a reflective statement.
2. `gui/handlers.py:_run_insight_mode` and `core/insight/synthesizer.py`: recent
   conversation is absent and every pattern turn gets report instructions.
   Feed bounded recent exchanges to synthesis; separate presentation from
   evidence gathering. Failed analysis is unavailable evidence, not zero events.
3. `core/insight/provenance.py`: mixed User/Assistant documents are labeled
   entirely as user statements. Split speaker spans before clipping; preserve
   source identity. Generated vault metadata is discarded by the sweep.
4. `memory/corpus_manager.py:search_keyword`: searches merged attachment text as
   the user side even when separate `user_text` exists. Preserve general
   attachment lookup, but make evidence sweeps request the authored side.
5. `core/prompt/gatherer_knowledge.py:_upload_is_live`: every fresh upload gets
   admitted regardless of the current topic. Require a relevance match or a
   document continuation for freshness to bypass the relevance threshold.
6. Conversation grounding instructions: old replies and generated notes are
   treated as authority; rest becomes earned productivity and explanations
   assert unsupported recovery mechanisms. Clarify evidence ownership and
   uncertainty while preserving useful, balanced decision support.

## Acceptance and contingencies

- Synthetic versions of all five turns exercise actual routing, sweep,
  provenance, upload admission and synthesis prompt functions.
- Explicit requests for analysis/documents continue to work; advice remains
  substantive and can assess either side without prescribing a personal dose.
- No inference that a planner failure means a corpus has zero observations.
- Tests and lint run in bounded batches. A failure in a shared contract is
  investigated before broadening changes; no expensive full-suite process.
- Live provider or restarted-Daemon checks that cannot run locally remain
  explicitly pending in the final handoff. No claim of measured latency or
  end-to-end quality improvement without a corresponding probe.

## Additional findings to bring to the owner

- Broad keyword facets (such as “work”) collect irrelevant philosophical and
  technical passages. Source hygiene is necessary but relevance/ranking also
  needs a separately measured retrieval evaluation.
- Temporal narrative contradicts newer conversation (completed social plans
  remain pending; assignment scope and names drift). Existing streak warnings
  quote the discarded claim back into context and project continuity without
  observing each intervening day.
- Profile and graph contain malformed, stale or misattributed facts. Existing
  curation must be assessed read-only before any repair is applied.
- The short clarification pays for many independent retrieval sections and a
  planner; “recall” labels also misclassify updates. Routing/retrieval should
  use resolved conversational intent, with regression data before changing
  broad thresholds or retrieval budgets.
- Insight mode has no phase timings in the supplied dump. Its stream watchdog
  checks elapsed time only after a chunk arrives, leaving a silent stream
  unbounded; investigate and cover this deployed path.

## Fable referee pass (2026-09-06, after the Codex batch)

Verified: ordered-slice guard, budget-meter parity, tool-wiring parity and
model-capability guards green; ruff clean on every touched file; an 838-test
targeted batch (insight, gate, planner, uploads, fact_source, corpus keyword,
generate paths, handlers, sep04–sep06 suites) green.

Fixed on top of the batch:
- `history[-8:]` in `recent_conversation_context` allowlisted with reasoning
  (UI history is oldest-first; the corpus fallback already reverses).
- `test_api_error_storage_guard` boundary test read `call.kwargs["metadata"]`;
  the deployed `add_conversation_memory` call is positional.
- `label_evidence` runs twice (sweep `_finalize` + handlers). It was not
  idempotent: the quoted-correspondence split re-suffixed doc_ids. Guarded.
- The live probe crashed inside `ModelManager.generate_async`: both
  local-model branches forwarded `**kwargs` beside `model_name=target_model`.
  The per-call `model_name` is now popped once at the top. The probe now
  selects its model with `switch_model` (a bare manager has no active model).

Live probe (kimi-3, three bounded calls, nothing stored): all three replies
matched the accepted direction — conversational answer for the failed
analysis, balanced NICE-grounded weighing without a personal schedule, a table
only when a report was requested. Assistant interpretation [E2] was named as
such in every case. Raw stream output ends with the kimi-3 trailing `e`; the
deployed insight path strips it.

Pending for the owner:
- Decide whether `CONTEXTUAL_GROUNDING` (~440 tokens, incident-shaped wording)
  belongs in every turn's cached prefix or only in the insight/medical paths.
- Restart the daemon (pid 1221038 predates this batch) and commit.
- Read-only assessment of the profile/graph/narrative findings listed above
  before any repair; no store change was made in this audit.

## Live retest and root causes (Fable, 2026-09-06 afternoon)

Three designed turns after the 14:02 relaunch. Every defect below is fixed in the
working tree (plan: `~/.claude/plans/tidy-hopping-kahan.md`; changelog entry
"2026-09-06 generalization + live-test root-cause batch").

| Symptom | Root cause (verified) | Fix |
|---|---|---|
| `Homework4.docx` (208 d old) on a personal turn | `reference_docs_manager._keyword_search`: `section in query_lower` with `section=''` → 0.9 on every query; keyword slots fill first | `_containment_match`; keyword scores never satisfy the relevance leg; upload relevance marker |
| Turn 2: 14 PubMed abstracts, 36 personal items omitted → "no personal data" | research prepended, unclipped, 12K cap in list order | `evidence_layout.layout_evidence` (personal-first, external ≤25%), `external_snippet_chars` |
| PubMed junk (pitchers, nurses) | `_PUBMED_STOPWORDS` deleted "medication"; anchor = first surviving term | domain nouns un-stopped; every declared axis mandatory; synonyms wired |
| Pattern channel "insufficient" | planner literal `"null"` end survived freeze | sentinel normalization, open end → today, per-phase drop |
| Turn 3 went to agentic memory, not insight | detector lacked noun "analysis"/"what my record establishes" | `_RECORD_ESTABLISHES_RE` + operation nouns |
| 45 s decision rounds | decision call sent no reasoning key (kimi-k3 reasons by default) | `generate_once(disable_reasoning=True)`; native-tools registry derived |
| "Give may refer to:" wiki stubs | live-API fallback: raw word split, `is_disambiguation=False` hardcoded | shape check on every extract; stopword-aware term selection |
| `took=5 mg … (said: "I did not take …")` | no negation-scope check; affirmative clause lost ownership to a later "his" | clause-scoped negation + ownership |
| rest framed as earned by yesterday's output | grounding text was incident-specific and universal | `UNIVERSAL_GROUNDING` + conditional `DECISION_SUPPORT_GROUNDING`, generic wording |
| STM "recall" on a fresh self-report / a request | prompt biases to recall; no data-token or request check | `new_data_override` → `unclear` |
| 11K tokens for a one-line status update | no shape-based budget; `max_facts` dead | `SELF_REPORT_RETRIEVAL_TRIM`, live `max_profile_tokens` |

Additional findings: sweep relevance → `scripts/probe_insight_sweep_relevance.py`
(owner grades the JSONL before any threshold change); narrative completed plans →
`utils/completed_plan_claims.py`; profile/graph → Curation Center scan (owner).

Pending: owner restart, rerun the three queries (turn 3 is kimi-3's first
native-tool run), two commits (`commit_message.txt`, then `commit_message_2.txt`).

## OWNER-GATED ACTIONS (checklist — nothing below is done by an agent)

- [ ] Restart the daemon (the 14:02 instance, pid 1239997, predates this batch).
- [ ] Rerun the three test queries in order and review the debug dumps
      (turn 3 = kimi-3's FIRST run on the native tool protocol — if tool calls
      misbehave, the change to gate is `protocols._native_tool_registry`).
- [ ] Commit 1: `commit_message.txt` (Codex conversation-audit batch + referee).
- [ ] Commit 2: `commit_message_2.txt` (generalization + live-test root causes).
- [ ] Run `python scripts/probe_insight_sweep_relevance.py --last 10` (read-only)
      and grade the JSONL under `eval/runs/` before ANY sweep threshold change.
- [ ] Curation Center → Scan (queue mode) for the profile/graph read-only review;
      export the queue for frontier review.
- [ ] Still open from earlier batches: Outlook/Azure app registration; the
      `data/*_candidates_2026090[35]*.txt` files not yet applied.

### Retest 15:10–15:2x observations (build with the batch)
- Turn 1: routed to the agentic MEMORY arm — the LLM trigger flipped to
  `needs_memory_search` on the identical text; the new self-report predicate
  missed because the pasted query was hard-wrapped (3 lines) and tripped the
  paste guard. Both fixed (soft wraps = one message; bare self-report with no
  recall cue stands the LLM memory verdict down; `test_gate_self_report_memory_backstop.py`).
  Needs a retest after restart.
- Turn 2: insight-mode, 33/33 rendered items personal, 0 external, no omission,
  balanced prose. Pattern channel STILL insufficient: the planner emitted ZERO
  phases this run ("event_phased requires at least 2 phase(s)") — a planner
  output gap, not the `"null"` end case. FIXED same session (`deliberation.structural_phases`, freezes READY with a note): when `analysis_kind ==
  event_phased` arrives with no phases, derive a deterministic two-phase split
  from the request window / most recent dated event before declaring the
  channel insufficient (the `deterministic_fallback_spec` ladder exists; it did
  not engage here).
- Turn 3 (same retest): insight-mode + report presentation + no wiki stubs
  (detector/wiki fixes held). Evidence quality: the user's own prior requests
  to Daemon, a junk fact, lecture text and venting rode in on the broad theme;
  the report guessed drug identities and assigned labels the user never used.
  FIXED: `exclude_assistant_directed_items`, `fact_triple_is_junk`, synthesis
  rule 2b (as written / no labels). Theme breadth itself → grade the probe
  output (owner) before touching sweep thresholds.

### Retest 15:37 (build with all fixes) — turn 1
- Route correct: gate "no trigger", enhanced, no uploads, no wiki, no decision
  block, profile 60 → 24 facts. Prompt still ~11K: [RECENT CONVERSATION] n=10
  = 4.8K tok (long test replies) and [PROJECT COMMIT HISTORY] n=5 = 2.4K tok.
  FIXED: self-report trim now also zeroes git commits and caps recent turns at 6.
- Reply again justified rest by tallying yesterday's output ("shipped nine
  fixes… the ledger's more than fine"), mirrored from Daemon's OWN earlier
  replies in recent conversation. Prompt-only lever tightened (universal rule
  names the pattern). OBSERVATION for the owner: this is the model copying its
  prior phrasing; a deterministic check would need a response-shape detector
  for "earned/ledger/deserve" justifications — not built (would be vocabulary).

### Retest 15:37 — turn 2 (structural phases live)
- Pattern channel ran (ready, 2 phases, corpus/notes/pubmed/pattern all
  succeeded). Regressions: 9 of 58 items rendered (layout walker + renderer
  stopped at one oversized unclipped item) and 4 surgical PubMed abstracts
  (anchor keys arrived as `rest_days`/`medication_use`; outcome axes absent
  from query terms). FIXED: skip-and-continue in both walkers, pre-layout
  clip, underscore-aware anchors, missing axes become mandatory groups.
  Needs a retest after the next restart.
- Note: the six "outcome events" the pattern engine matched (a supplement/covid
  turns) came from the planner's indicator vocabulary ("feeling better",
  "experiencing symptoms") — planner-quality, logged for the probe review.

### Retest 15:57 (full build) — turns 3 and 2
- Turn 3: insight-mode report, quoted/dated/strength-rated, no guessed names
  or labels, 41/47 rendered. Turn 2: pattern channel ran, 35/47 rendered,
  personal records first, balanced prose.
- FIXED after this pass: anchors now reach the per-rung PubMed ranker via the
  adapter chain (they never did in production); exclusions re-run after the
  phase-event merge ([E1] was the earlier request); junk-subject and
  bare-quantity facts filtered; greeting/ack turns excluded.
- Left as planner-vocabulary review: outcome word "symptoms" matched covid/
  a supplement turns as outcome events (valid under before/after, but broad).
- Final state: every code-closable finding from all three turns is in the
  tree. Next restart picks up: post-merge exclusions, PubMed anchors in
  production, junk-subject/greeting filters, extended self-report trim.

### Retest 16:10 — turn 2 on the final build: VERIFIED
Pattern channel ran (structural phases), 32/33 rendered, personal records
first, no request text as evidence, PubMed 11 junk → 3 loose matches with the
channel marked `no_relevant_results`. Note: the synthesis tail carried the
elevated-tone TONE GUARD — session tone carry-over, not this query's tone.
Commit: the two batches overlap in files; use `commit_message_combined.txt`
for one commit (`git commit -F commit_message_combined.txt`).
