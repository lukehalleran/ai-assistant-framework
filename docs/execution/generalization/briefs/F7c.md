=== F7c: the remaining knowledge gatherers return a typed failure instead of an empty section (git, proposals, skills, graph, threads, narrative, self-notes, emails) ===
(Durable copy, re-created in the repo on 2026-09-14 after a machine crash wiped the /tmp scratchpad. Content unchanged; only the rules path is now explicit.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Parent review amendments" → "F7 split and gatherer outcome shape (parent decision, 2026-09-14)", F7c row and "Limits recorded now".
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md. This batch answers anchors #75, #76, #77, #78, #79, #81, #82, #83 and #84.
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-007-3.md.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL, including "Response file", SHELL GUARD, NON-PYTEST CODE, GIT INDEX AND PYTEST HYGIENE, NON-UNIT TESTS, INTERRUPTION and MEMORY.

COMPLIANCE (added 2026-09-14 after the F7a and F7b workers broke process rules; read this twice):
- Your FIRST source-file write must come AFTER the failing-first run is recorded in batches/F7c.md.
  - F7b edited sources before failing-first, and it cost three revert cycles.
  - Do not open core/prompt/gatherer_knowledge.py with Edit or Write until §"failing-first" is in the packet.
- Run NO non-pytest Python of any kind: no `python -c`, no `python3 -`, no heredoc (even an empty one), no REPL, no script.
  - F7a and F7b both broke this.
  - To inspect scan JSON, use `grep`/`jq`.
  - For ground truth about behaviour, write a pytest test.
  - The only exceptions are `python -m ruff`, `python scripts/check_bug_classes.py scan`, and the single `import utils` sanity check.
- Before writing a new test's fixture input, check it against the input gates it must pass. Two known traps:
  - `_should_skip_wikipedia` matches conversational patterns by raw substring;
  - builder queries of ≤4 words read as fragment continuations and zero some sections.
  - A control that fails on the unedited source means your fixture is wrong, not the source.
- Any breach must be disclosed in the packet immediately, and the parent records it.

MANIFEST CHECK (before any edit; if either check fails, stop and report)
Run from the checkout root with S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:
  (a) `sha256sum -c --quiet $S/manifest_post_F7b.txt` must print nothing and exit 0.
  (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F7b.txt` must print nothing.
GUARD: run the /proc/comm pytest guard before EVERY pytest command, `--collect-only` included. If another pytest is running, wait in the foreground and re-check about every 60s. Never start pytest in the background.
MEMORY: run tests in the FOREGROUND, in chunks of ≤9 files. Before each chunk, MemAvailable must be ≥4000; otherwise wait and re-check.
DATA NOTE:
- Before the first pytest and AFTER EVERY CHUNK, record:
  - `ls -la --time-style=full-iso data` (top level only);
  - `ls -ld --time-style=full-iso logs`.
- Compare against the parent's post-F7b baseline in batches/F7b.md.
- If anything appears or changes, STOP and report which run did it. Never delete anything.
SCRIPTS: pytest only.
- `python -c`, `python3 -`, REPL, heredoc snippets (even empty ones) and throwaway scripts all need parent approval BEFORE they run.
- ruff, the read-only scan and the one `import utils` sanity check are the only exceptions.
- File edits use the Edit or Write tools only. Never leave stray files in the repository.
- No real git subprocess, Chroma store, graph, email/Gmail service, embedder, model or network. Use fakes.
ORDER REMINDER: print `sha256sum core/prompt/gatherer_knowledge.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old flattening may be repaired only with the new assertion plus a paired control. List every existing-test edit. Anything else is a STOP with an escalation packet.

PARENT-VERIFIED FACTS
- Line numbers are from the post-F4 tree; F7a and F7b edit other methods in the same file, so RE-LOCATE by def name and the quoted code.
- POST-F7b RE-LOCATION (parent-verified on the integrated F7b tree, gatherer_knowledge.py `18390202…`):
  - **Before the wiki/semantic methods (+23, from F7a only):**
    - `get_git_commits` 1023 (except warning "Failed to get git commits" 1115);
    - `get_proposed_features` 1118 (warning 1170);
    - `get_procedural_skills` 1173 (warning 1213);
    - `get_graph_context` 1216 (warning "Graph context retrieval failed" 1275);
    - `get_unresolved_threads` 1278 (warning 1302).
  - **After the wiki/semantic methods (+36, from F7a and F7b):**
    - `get_narrative_context` 2006 (`narrative = corpus.get_narrative_context()` 2026; warning "Failed to retrieve narrative context" 2035);
    - `get_daemon_self_notes` 2038 (annotation import 2088 / call 2093; outer debug "daemon_self_notes retrieval failed" 2100);
    - `get_relevant_emails` 2133 ("Email relevance filtering failed" 2290 = #83; "Email retrieval failed" 2296 = #84).
  - **Builder caller:** `narrative_state = self.context_gatherer.get_narrative_context()` is at core/prompt/builder.py:1072.
  - **Integrated state of the other methods:** F7a's and F7b's methods in the same file are integrated and read-only for you.
- The import `from utils.retrieval_outcome import OutcomeList, outcome_status` exists after F7a (line 65 on the F7a tree); confirm it.
- TEST EXCLUSION (added after F7a): NEVER run tests/unit/test_graph_integration.py. It writes data/user_profile.json through the ContextGatherer → UserProfile() fallback (see R_common_rules.md NON-UNIT TESTS). Before running any file that builds `ContextGatherer` with a coordinator lacking `user_profile`, or calls `UserProfile()` without a patched path, exclude it. `data/user_profile.json` (598 bytes) exists and must stay untouched; include it in your data/ baseline.
- `get_git_commits` (def 1000):
  - `if not GIT_MEMORY_ENABLED or limit <= 0: return []` (1018–1019);
  - repo-status branch: `GitMemoryExtractor.extract_commits` (1029). It swallows FileNotFoundError/OSError/TimeoutExpired and non-zero returncode to `[]` (knowledge/git_memory.py:104–110); that is a read-only sibling;
  - index branch: `if not chroma or 'procedural' not in chroma.collections: return []` (1044–1045); `chroma.get_recent('procedural', …)` (1052) and `chroma.query_collection('procedural', …)` (1055) raise (no swallow);
  - except 1091–1093 → `return []`: ANCHOR #75.
- `get_proposed_features` (def 1095):
  - `if not CODE_PROPOSALS_PROMPT_ENABLED: return []` (1111–1112);
  - `self._proposal_filter.get_proposals(query, limit=limit)` (1127) → core/prompt/proposal_filter.py:610 → memory/proposal_store.py `query_proposals`, which swallows to `[]` (161–163); a sibling for F11b;
  - except 1146–1148: ANCHOR #76.
- `get_procedural_skills` (def 1150):
  - `if not PROCEDURAL_SKILLS_ENABLED: return []` (1165–1166); `if not hasattr(self.memory_coordinator, 'get_skills'): return []` (1168–1169);
  - `self.memory_coordinator.get_skills(query, limit=limit)` (1171) → memory/memory_retriever.py `get_skills`, which swallows (1179–1181); a sibling for F10/F12;
  - except 1189–1191: ANCHOR #77.
- `get_graph_context` (def 1193):
  - `if not KNOWLEDGE_GRAPH_ENABLED: return []` (1208–1209); `if not graph or not resolver or graph.node_count() == 0: return []` (1214–1215);
  - `extract_graph_entities` (1220) and `graph.get_context_sentences(…)` (1224) raise (no swallow);
  - except 1251–1253: ANCHOR #78.
- `get_unresolved_threads` (def 1255):
  - `if not THREAD_SURFACING_ENABLED: return []` (1268–1269); `hasattr` gate (1271–1272);
  - `self.memory_coordinator.get_unresolved_threads(max_results=…)` (1274) → memory/memory_coordinator.py:585, which swallows (598–600); a sibling for F11a;
  - except 1278–1280: ANCHOR #79.
- `get_narrative_context` (def 1970, SYNC, returns str):
  - `if not NARRATIVE_CONTEXT_ENABLED: return ""` (1983–1984); no corpus manager → `return ""` (1996);
  - `corpus.get_narrative_context()` (1990) → memory/corpus_manager.py:605, which swallows to "" (659–661); a sibling for F12;
  - except 1998–2000 → `return ""`: ANCHOR #81;
  - its ONLY caller is core/prompt/builder.py (≈1075 post-F5; `narrative_state = self.context_gatherer.get_narrative_context()`), inside a try whose except logs at debug level. F5 records the narrative outcome there (exception → failed). Re-read batches/F5.md.
- `get_daemon_self_notes` (def 2002):
  - `if not chroma: return []` (2015–2016); `if not results: return []` (2024–2025);
  - `chroma.query_collection("daemon_self_notes", …)` (2019) raises (no swallow);
  - an inner per-item try 2047 / except 2058 swallows annotation failures and continues; leave it and record it as a sibling;
  - outer except 2063–2065: ANCHOR #82.
- `get_relevant_emails` (def 2097):
  - legit early returns: `except ImportError: return []` (2118–2119); `if not EMAIL_PASSIVE_CONTEXT_ENABLED` (2121–2122); distress suppression (2126–2127); no cue/contacts (2146–2148); `if not messages: return []` (2165–2166);
  - `service.search(…)` (2159) → core/email/service.py `_fan_out` swallows per-provider errors (a sibling);
  - the contact resolution local guard (2185–2186) stays;
  - inner scoring try 2193 / `except Exception` 2253 → "Fail closed rather than injecting arbitrary inbox content" → `return []` (2257): ANCHOR #83;
  - outer except 2259–2261: ANCHOR #84.
- Existing tests that reference these methods (parent grep):
  - tests/unit/test_sep12_repository_status_context.py (git; `test_git_timeout_does_not_substitute_stale_index` asserts `== []`; also narrative);
  - tests/unit/test_proposal_filter.py;
  - tests/unit/test_independent_prompt_audit.py;
  - tests/unit/test_narrative_staleness.py;
  - tests/unit/test_sep10_probe_dump_actions.py;
  - tests/unit/test_sep10_probe_dump_interpretation.py;
  - tests/unit/test_email_passive_context.py (`test_embedding_failure_does_not_inject_unranked_email` asserts `out == []`);
  - tests/unit/test_sep03_followups_gating.py;
  - non-unit tests/test_thread_surfacing.py (threads; F5's parent vetted it as fake-only; still read it and record the decision under NON-UNIT TESTS rules).

OWNERSHIP
- core/prompt/gatherer_knowledge.py: the bodies of the eight methods above ONLY.
- New tests/unit/test_gatherer_outcomes_remaining_knowledge.py.
- New docs/execution/generalization/batches/F7c.md.
- The response file named above.
- Read-only: every other method in the file, memory/**, knowledge/git_memory.py, core/prompt/proposal_filter.py, core/email/**, core/actions/**, builder, formatter, orchestrator, handlers, utils/retrieval_outcome.py, config/**, docs/execution/generalization/briefs/**, and everything else. Every class-guard-owned path is also read-only (see R_common_rules.md "Never edit").

CONTRACT
1. Anchors #75, #76, #77, #78, #79, #82 and #84: the except returns `OutcomeList.failed(type(e).__name__)` instead of `[]`, and the existing log line stays byte-identical.
2. Anchor #83 (email ranking): return `OutcomeList.failed("relevance_unavailable")`. It still carries NO items: fail-closed is preserved, and the comment stays.
3. Anchor #81 (narrative, str):
   - the except keeps its warning log and then re-raises (`raise`), so the builder's existing try catches it and F5 records `narrative` as failed.
   - The disabled flag and the missing corpus manager still return "".
   - Prove with a builder-level test that the prompt still builds and `narrative_state` is still "" on failure.
4. Every legit early return listed above stays unchanged (no_results / not attempted).
5. Producer-internal swallows are READ-ONLY siblings. Record each with its file:line and owning batch; do not edit them: git_memory `extract_commits`, proposal_store `query_proposals`, memory_retriever `get_skills`, memory_coordinator `get_unresolved_threads`, corpus_manager `get_narrative_context`, email service `_fan_out`, and the self-notes annotation loop.
   - Consequence: until F9–F12, a failure inside those producers still reads as no_results. State that as a limitation per section.
6. Privacy: reasons are constant labels or exception class names only.

TESTS (tests/unit/test_gatherer_outcomes_remaining_knowledge.py; fakes only; parametrize to stay within size)
- FAILING FIRST in one command: `sha256sum core/prompt/gatherer_knowledge.py`, then the new tests on the UNEDITED source. List the failures.
- For each of git (index branch: raising chroma), proposals (raising proposal filter), skills (raising `memory_coordinator.get_skills`), graph (raising `get_context_sentences`), threads (raising `memory_coordinator.get_unresolved_threads`) and self-notes (raising `query_collection`), through the deployed method with the feature flag enabled:
  - `outcome_status(result) == ("failed", "<ExcClass>")` and `result == []`;
  - control: a healthy empty dependency → no_results;
  - control: healthy non-empty → today's items.
- Emails:
  - a raising embedder → failed / relevance_unavailable and `result == []`;
  - an exception in the outer block (e.g. a raising `service.search`) → failed / class;
  - controls: no cue → `[]` / no_results; healthy ranked messages → today's output.
- Narrative:
  - a raising `corpus.get_narrative_context` fake → the gatherer re-raises;
  - through `full_builder`: the prompt still builds, `narrative_state` is "", and `_section_outcomes["narrative"]["status"] == "failed"`;
  - control: disabled → "" and no narrative entry per F5.
- Privacy: a distinctive marker in the query and the exception messages never appears in any reason.
- Focused: the new file plus the 8 unit files above (one chunk of 9). tests/test_thread_surfacing.py runs only under NON-UNIT TESTS rules.
- Sweep:
  - every remaining tests/unit importer of core.prompt.gatherer_knowledge / KnowledgeRetrievalMixin / core.prompt.context_gatherer (grep and list), in chunks of ≤9.
  - Never run tests/test_web_search_manager.py or tests/test_prompt_internal_methods.py.

SCAN: pre- and post-edit read-only scan. Expect the dm18 rows for #75–#79 and #81–#84 (9 rows) to go STALE and new = 0. A re-raising except is not a dm18 finding; confirm it from the scan.
RESPONSE FILE:
- Include the R_common_rules contents per anchor.
- Record the siblings table (item 5) with owning batches.
- List #85–#92 as F8 (-4).
- Record the CGR-007 status: which anchors are answered by -1/-2/-3.
SIZE: target ≤400 changed lines; HARD stop at ≤450. Stop before exceeding it and return a split proposal; the natural split is emails plus narrative vs the rest.
ORDER: manifest → create F7c.md → pre-edit scan → tests → failing-first (digest in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
