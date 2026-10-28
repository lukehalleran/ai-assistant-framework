# SITUATION_COVERAGE_AUDIT_20260901.md — Situation-Specific Feature Sweep

*2026-09-01. Four-lens parallel agent sweep (display/storage transform parity ·
narrow trigger/recovery shapes · protocol/mode parity · store/read hygiene
parity), headline findings hand-verified. THE CLASS: a mechanism built for the
one situation where the need was first observed, while sibling situations with
the identical need stay uncovered. Canonical examples (all previously fixed):
the stream-artifact strip that covered storage+enhanced display while
agentic/raw/duel yielded raw text (08-14); the third summary write path
bypassing sanitize (08-22); XML action forcing native-only (08-29); the
private-sphere guard born coursework-only (09-01).*

Verdicts: **FIX** (executed this session), **DESIGN** (real gap, needs a
frontier design session — do not hand to a cheap executor), **DELIBERATE**
(documented scope decision — leave), **REJECTED** (agent claim wrong on
inspection).

## FIX — executed 2026-09-01 (batch C: hygiene parity, batch D: stream parity)

| # | Item | Where | Gap |
|---|------|-------|-----|
| C1 | **Insight sweep applies ZERO hygiene** | `core/insight/sweep.py` chroma loop | "UNGATED by design" was implemented as no-cosine-gating AND no junk/quarantine/supersession filtering — API-error docs, quarantined items, and superseded facts flow into evidence for the highest-stakes synthesis mode. Ungated ≠ unhygienic: corpus channel already filters via `search_keyword`; the 6 chroma collections filter nothing |
| C2 | **memory_expander applies ZERO hygiene** | `memory/memory_expander.py` all 3 fetch paths | agentic expand_memory "zoom in" surfaces junk/quarantined/superseded docs the primary retrieval sites all filter |
| C3 | **Backups miss every store added since 07-14** | `utils/backup_manager.py::backup_targets` | adaptive_exemplars, learned_relations, tone_state, pending_actions, curation_queue (+ narrative staleness flag) are atomic-write stores absent from the backup set — restore loses all learned/derived state |
| C4 | `_store_summary` partial sanitize | `memory/shutdown_processor.py` | has artifact-strip + junk check (08-22 fix) but not the full `sanitize_for_storage` (thinking-leak + degenerate defenses) the primary storage path gets |
| D1 | doc-gen / self-note text never artifact-stripped | `gui/handlers.py` doc-gen + self-note finals; `knowledge/document_generator.py` generate_once outputs | the only display/storage paths still missing `strip_trailing_stream_artifact` (kimi-3 trailing-'e'/`<|sep|>` class) |
| D2 | **Degenerate-stream watchdog insight-only** | `gui/handlers.py` enhanced + agentic stream loops | the 08-31 runaway-garbage watchdog covers only insight synthesis; a kimi-3 loop on enhanced/agentic still displays garbage for minutes (storage IS protected by sanitize step 8 — display isn't) |
| D3 | Decision-timeout fallback web-only | `core/agentic/controller.py:~847` | the 75s-timeout deterministic fallback substitutes WEB searches only; memory-routed sessions fall to synthesis with zero context. Extended to memory-routed sessions (file/computation stay synthesis-bound — their params can't be inferred, deliberate) |

## DESIGN — real gaps, deferred to frontier design sessions

- **Inability-retry contract is web-search-only** (`_SEARCH_INABILITY_MARKERS`):
  the "assistant disclosed inability + user retries = deterministic re-run"
  contract logically covers file/memory/calendar/computation inability.
  Generalizing needs a per-channel marker registry AND channel-correct routing
  (a file-inability retry must route to tools, not web) — design session.
- **Veto silence registry**: only TONE vetoes get the acknowledge+offer
  deferred-request loop; intent veto, private-sphere/temporal-generic
  suppression fail silently. Some silences are deliberate (anti-excavation);
  none are marked. Needs an explicit per-arm decision table before extending.
- **Narration/empty-output guards asymmetric across generators**: agentic
  final + decision-reuse are guarded (`narration_shaped_final`,
  `_usable_decision_answer`); enhanced streaming, best-of, doc-gen, insight
  synthesis have no narration guard. Enhanced retrofit means whole-bubble
  replacement mid-stream — design session.
- **Employer-logistics injection**: `get_user_anchors()` now resolves
  employer/orgs (09-01) and feeds the private-sphere guard, but there is no
  `apply_employer` term backstop or employer line in the trigger prompt
  (school has both). Extend when a live need appears.
- **API chat_service sanitization parity** (needs verification): handler yields
  are sanitized before chat_service sees them; the extra `_strip_xml_artifacts`
  is belt-and-suspenders. Verify with a live SSE capture before calling it a gap.

## DELIBERATE — documented scope decisions, leave

- Completed-resend serve guards (300s / completed-only / action-excluded /
  user-saw-it): four documented layers, complete as designed.
- `extract_rare_proper_nouns` TitleCase-only: under-fires by design; lowercase
  rare terms (medications, products) are a known non-goal for anchoring.
- Grounding check exempting duel/doc-gen "in v1"; raw mode bypassing guards.
- Uncertainty detector OFF / review gate LOG-ONLY (2026-08-28 telemetry
  verdicts — do not revive without new evidence).
- Pending-proposal one-shot TTL; XML `tool_choice` absence (protocol-structural,
  mitigated by force-prompt + no-marker retry + parse-time validation);
  file/computation timeout fallback absence (params not inferable).
- External-knowledge ingestion (wiki/obsidian/git) skipping conversation junk
  filters — different data class.

## REJECTED — agent claims wrong on inspection

- "Insight sweep should skip appraisal graph edges": including them LABELED
  (`is_appraisal=True` → provenance renders "your words at the time") is the
  stance doctrine working, not a leak. Perspectives are evidence in insight mode.
- "XML forced-action parity unfixed": the 08-29→09-01 fixes (XML force prompt,
  generic attr parse, no-marker retry, parse-time accepts_params/accepts_check)
  are the mitigation; remaining asymmetry is protocol-structural (above).
- "`add_to_collection` needs universal guards": wrong layer — it's the generic
  primitive; guards belong in typed wrappers, and the summaries/conversations
  callers are now all guarded (C4 closed the last one).
- "Correction detectors not called on handler modes": `run_post_response_detectors`
  is invoked from `handlers._write_turn_telemetry` (08-24 fix) — agent missed
  the call chain.

## Status

FIX rows executed 2026-09-01 (two cheap-executor batches + frontier review);
DESIGN rows are the standing worklist for future sessions.
