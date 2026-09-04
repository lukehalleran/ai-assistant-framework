# Codex audit sweep — 2026-09-03

Status: code changes are **uncommitted**. No live-store or `data/` writes were
performed. Fable's existing edits were preserved. This note is the resume point
for the next session; see `AUDIT_SWEEP_20260831.md` for the preceding full sweep.

## Coordination / interrupted Fable batch

Fable's first 11-file cat-session correction was already present when this audit
started. Fable then launched three follow-up groups and ran out of session credit.

- Continuity/topic/STM group **landed**. Its new regression file initially had
  two faulty tests: one expected `recall` even though the message directly
  answered the preceding question (the existing deterministic rule correctly
  says `clarification`), and one constructed `ContextPipeline` via `__new__`
  without `_stm_max_recent`. Codex corrected those tests. The changed continuity,
  STM, cat-session, and graph integration paths pass together.
- Graph read-model + hub-barrier group **landed**. The relation-keyed edge index
  is now authoritative while the NetworkX `DiGraph` remains a topology index.
- Gating/role/daily-note-timer group **did not land**. There are no partial edits
  to unwind. Fable should resume this group from its saved plan after 18:00:
  passive email/pet filtering, three-day upload freshness, casual retrieval
  zeroes, personal-note threshold/config, occupation-vs-role normalization, and
  the lightweight daily-note entrypoint/systemd templates.

Do not run Fable's owner-gated cleanup scripts or touch live stores until the
daemon is down and Luke explicitly approves the dry-run output.

## Confirmed bugs fixed by this Codex sweep

### Prompt summarization

- `LLMSummarizer` could stringify an async-generator object as a summary for API
  models. Reflection generation had the inverse failure: a local-model string was
  treated as an async iterator and silently produced no reflections.
- Internal generation now prefers `ModelManager.generate_once`, supports legacy
  string/content/dict/async-stream results, and applies the timeout to stream
  consumption as well as request creation.
- Replaced a test that caught every exception and therefore could never fail with
  a real persistence assertion.

### API and email

- Cross-provider and direct-provider email sorting compared ISO strings, so equal
  instants with different timezone offsets could be ordered incorrectly. Sorting
  now uses parsed instants and places malformed timestamps last.
- Email cache objects were returned by reference, allowing a caller to mutate the
  shared cache. Cache reads/writes now use defensive copies.
- Negative email limits/windows and curation activity limits no longer invoke
  inconsistent negative-slice/provider behavior.
- Uploads were fully buffered before the aggregate 100 MB check; a later rejected
  file leaked prior registrations and temp files. Uploads are now streamed in
  bounded chunks with transaction-style rollback, including client cancellation.
- A disconnected API chat client could leave the shielded next-chunk task running
  against session state. The task is now cancelled/awaited and the generator is
  closed in `finally`.

### Memory/eval robustness

- Cross-collection dedup used bare document IDs even though IDs are only unique
  within a Chroma collection. Delete/mark guards now use `(collection, id)`.
- Snapshot `from_dict` methods mutated caller-owned dictionaries by popping
  nested fields. They now copy first.
- Curation scans assumed perfectly aligned Chroma arrays. Missing documents and
  metadata now degrade to `None`/`{}` instead of mis-indexing or crashing.
- Best-effort Wikipedia gating only caught selected exception types; unexpected
  client failures could break an otherwise viable gate result. It now fails soft
  for ordinary exceptions (while cancellation remains uncaught).
- Removed unconditional user-profile debug prints from normal runtime.

### GUI output safety

- Synthesis-card fields were interpolated into `gr.HTML` without escaping.
  User/LLM-derived strings are now HTML-escaped and covered by a malicious-value
  runtime regression. No second uncontrolled dynamic `gr.HTML` path was found in
  `gui/launch.py`.

## Verification completed

- `ruff check .`: clean.
- `compileall` across API/core/eval/GUI/knowledge/memory/models/processing/scripts/
  utils/web: clean.
- Fable changed-path continuity + cat/graph integration: **138 passed**.
- Codex memory/email/GUI/summarizer focused group: **89 passed**.
- Upload rollback/cancellation + negative curation limit: **3 passed**.
- Agent-branch suite: **129 passed, 22 deselected** (rerun outside the socket-
  restricted sandbox after the sandbox-only failures were identified).
- Web production build: passed. Vite reported a roughly 1.07 MB main JS chunk.
- `git diff --check`: clean at the last verification point.

Some large combined pytest invocations finish their assertions but do not exit;
the same files complete normally when split. This is recorded below rather than
misreported as a green full-suite run.

## Obvious improvement areas found (not patched)

1. **Validate parseable profile structure on load.** Invalid JSON is already
   quarantined, but valid JSON with the wrong shape can fail at startup or later
   with `AttributeError`, `KeyError`, or `TypeError`. Add top-level object and
   required-container validation. Decide separately whether parseable-but-invalid
   files should be quarantined automatically.
2. **Define successful API-upload lifetime.** Failed/cancelled batches now clean
   up, but successful temp files and `_uploads` registrations have no TTL,
   per-session purge, or explicit delete endpoint. Choose whether file IDs are
   reusable, then add a janitor consistent with that contract.
3. **Fix pytest resource teardown.** Several broad combinations hang after many
   completed assertions, while isolated files exit promptly. Audit background
   executors/tasks and heavyweight prompt/memory fixtures; add suite-level leak
   detection so CI can distinguish a hang from a test failure.
4. **Reduce production log noise and exposure.** Token-budget, recent-memory,
   prompt-assembly, STM, and image diagnostics are emitted at WARNING, including
   user-query/STM material. Move routine diagnostics to DEBUG, redact/truncate
   content-bearing fields, and retain WARNING for actual degradation. Current
   logs become very large and obscure real failures.
5. **Split the web bundle.** The build succeeds, but the main chunk is about
   1.07 MB. Route/component lazy loading and explicit Rollup chunks would improve
   cold load and caching.
6. **Make external semantic-index absence operationally explicit.** Startup logs
   an ERROR because the configured external FAISS index/metadata paths are absent,
   then continues by design. Either provision/mount them, disable that feature in
   config, or downgrade the expected disabled state so real errors stand out.
7. **Treat E2B delete-404 as idempotent cleanup.** The sandbox can already be gone
   server-side; shutdown succeeds but the SDK emits ERROR for DELETE 404. Normalize
   this expected case in the wrapper if the SDK permits it.
8. **Dead-code cleanup (low priority).** Vulture still flags high-confidence
   candidates in prompt builder, eval snapshots, synthesis filter, memory expander,
   model manager, and gate system. Several are likely compatibility parameters;
   confirm call contracts before removal. Exact current output can be reproduced
   with `python -m vulture api core eval gui knowledge memory models processing
   scripts utils web --min-confidence 90`.

## Older deliberately deferred items still open

The prior sweep's owner/frontier deferrals remain the authoritative list:
F12 insight-branch runtime harness; F16 same-title/same-day calendar policy;
F25 canonical persisted-response asymmetry; F30 XML tool-teaching prompt budget;
F34 DNS-rebinding/SSRF transport pinning; and the F35 minor inventory. Re-evaluate
after Fable's current batch lands rather than changing storage or prompt contracts
in parallel.

## Suggested next sequence

1. Let Fable finish only its unlanded gating/role/timer group and run its named
   tests; do not redo the two groups already green.
2. Re-run Ruff, compileall, the three focused groups above, then the wider suite in
   small processes while investigating the teardown hang.
3. Review the full mixed diff as two conceptual commits: Fable's cat/continuity/
   graph work, then Codex's audit fixes. Human commits only.
4. Restart the daemon after review; runtime behavior still reflects the old
   in-process graph/thread/gating state until restart.

## Post-handoff resolution (Fable, 2026-09-03 evening/night)

Supersedes the coordination notes above where they conflict.

- **Group 3 (gating / role / daily-note timer) LANDED** by hand after the rate limit:
  passive-email config window + pet-name contact filter + `max_relevant_emails`/`max_graph_sentences`
  overrides (casual_social/emotional_support zeroes), uploads freshness 7→3 days, personal-notes gate
  0.45→0.60 with `min_results=0`, `role`→`occupation` synonym + cue family + few-shot rewrite,
  `scripts/daily_note_catchup.py` + `scripts/systemd/` templates (installed live; the 02:00 job now
  exits 0 with no tracebacks under the 3.11 env). Tests: `tests/unit/test_sep03_followups_gating.py`.
  The graph group also gained the hub barrier + seed-first ordering (`test_sep03_followups_graph.py`).
- **"Not patched" inventory:** items 1 (profile shape validation), 4 (log noise), 5 (bundle split:
  main chunk 1.07 MB → 21 kB app + vendor chunks ≤262 kB), 6 (FAISS absence = INFO disabled state),
  7 (E2B 404 idempotent) and 8 (three dead imports) are implemented — `tests/unit/test_codex_followups.py`.
  Item 2 (upload temp-file lifetime) is deferred as an owner deletion-policy decision. Item 3 (pytest
  teardown hang) did not reproduce: both `tests/unit` halves combined (3,324 tests) exited normally
  in 43 s; the hanging combination was not named, so it is left open.
- **One Codex change amended:** `core/prompt/summarizer.py`'s compatibility collector accepted a Mock's
  auto-created `.content` as text (`test_llm_summarize_recent_no_model_manager` went red); it now
  accepts only `str` content.
- Later the same night, two live probe turns verified the whole batch and produced eight more fixes
  (`tests/unit/test_sep03_live_probe_fixes.py`; narrative in the local CLAUDE_CHANGELOG.md).
- Verification after everything: 6,212 unit tests + 252 top-level tests, 0 failures; ruff clean;
  privacy hook scan of every added line clean.
