=== F7b: background-knowledge legs report failure, timeout and busy distinctly (wiki, semantic chunks, FAISS row reads) ===
(Durable copy, re-created in the repo on 2026-09-14 after a machine crash wiped the /tmp scratchpad. Content unchanged; only the rules path is now explicit.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Parent review amendments" → "F7 split and gatherer outcome shape (parent decision, 2026-09-14)", F7b row.
- "[verified, F4 parent review] A total row-read failure reads as `no_results`."
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md.
- This batch answers anchor #80.
- The semantic-chunks and `SemanticSearchIndex.search` row-read changes are BC-58 siblings recorded in the same response.
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-007-2.md.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL, including "Response file", SHELL GUARD, NON-PYTEST CODE, GIT INDEX AND PYTEST HYGIENE, NON-UNIT TESTS, INTERRUPTION and MEMORY.

MANIFEST CHECK (before any edit; if either check fails, stop and report)
Run from the checkout root with S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:
  (a) `sha256sum -c --quiet $S/manifest_post_F7a.txt` must print nothing and exit 0.
  (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F7a.txt` must print nothing.
GUARD: run the /proc/comm pytest guard before EVERY pytest command, `--collect-only` included. If another pytest is running, wait in the foreground and re-check about every 60s. Never start pytest in the background.
MEMORY: run tests in the FOREGROUND, in chunks of ≤9 files. Before each chunk, MemAvailable must be ≥4000; otherwise wait and re-check.
DATA NOTE:
- Before the first pytest and AFTER EVERY CHUNK, record:
  - `ls -la --time-style=full-iso data` (top level only);
  - `ls -ld --time-style=full-iso logs`.
- Compare against the parent's post-F7a baseline in batches/F7a.md.
- If anything appears or changes, STOP and report which run did it. Never delete anything.
NO REAL INDEX: `knowledge.semantic_search.get_index()` loads the REAL multi-GB FAISS index and parquet from data/. No test may trigger it: use a fake `SemanticSearchIndex` or monkeypatch `get_index`. No real Chroma, embedder, Wikipedia API or network.
SCRIPTS: pytest only.
- `python -c`, `python3 -`, REPL, heredoc snippets (even empty ones) and throwaway scripts all need parent approval BEFORE they run.
- ruff, the read-only scan and the one `import utils` sanity check are the only exceptions.
- File edits use the Edit or Write tools only. Never leave stray files in the repository.
ORDER REMINDER: print `sha256sum core/prompt/gatherer_knowledge.py knowledge/semantic_search.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old flattening may be repaired only with the new assertion plus a paired control. List every existing-test edit. Anything else is a STOP with an escalation packet.

PARENT-VERIFIED FACTS
- Line numbers are from the post-F4 tree. F7a edits earlier methods in the same file, so RE-LOCATE by def name and the quoted code.
- POST-F7a RE-LOCATION (parent-verified on the integrated F7a tree, gatherer_knowledge.py `5d79ddf2…`): every target below moved by +23 lines.
  - Import: `from utils.retrieval_outcome import OutcomeList, outcome_status` is present at line 65.
  - Wiki:
    - `_get_wiki_content` 1712; `_get_wiki_content_timed` 1728;
    - chroma in-flight guard `acquire` 1744;
    - chroma `except Exception` debug "falling back to API" 1822;
    - fallback except warning "Error getting wiki content" 1848 (its `return []` follows).
  - Semantic:
    - `_get_semantic_chunks` 1853; `_get_semantic_chunks_timed` 1864;
    - semantic in-flight `acquire` 1873; `return semantic_search_with_neighbors(query, k)` 1882;
    - similarity threshold filter 1909, with a second list rebuild (disambiguation filter) at 1919;
    - `chunks = list(chunks_by_title.values())` 1968;
    - timeout warning 1973.
  - knowledge/semantic_search.py is unchanged (`fe772c25…`): `row_data_map = self._read_rows(...)` 363, `if not data:` 370, `return OutcomeList(rows[:k])` 377.
- TEST EXCLUSION (added after F7a): NEVER run tests/unit/test_graph_integration.py. It writes data/user_profile.json through the ContextGatherer → UserProfile() fallback (see R_common_rules.md NON-UNIT TESTS). `data/user_profile.json` (598 bytes) now exists and must stay untouched; include it in your data/ baseline.
- Digests: gatherer_knowledge.py was `77d094d6…` before F7a; knowledge/semantic_search.py is `fe772c25…` (post-F4, re-verified after the crash).
- F7a added `from utils.retrieval_outcome import OutcomeList, outcome_status` to gatherer_knowledge.py. Confirm; if it is absent, add it.
- `_get_wiki_content` (wrapper, def ≈1689) is try/finally only. `_get_wiki_content_timed` (def 1705):
  - `if not query: return []` (1707) and `if self._should_skip_wikipedia(query): return []` (1711–1712): deliberate, no_results; leave them.
  - chroma leg, when `chroma` exists:
    - in-flight guard `if not _WIKI_CHROMA_INFLIGHT.acquire(blocking=False): … return []` (1721–1726);
    - try 1727, `chroma.query_collection(` (1733);
    - `except asyncio.TimeoutError:` sets `timings["timed_out"] = True`, warns "skipping wiki this turn", `return []` (1787–1797);
    - `except Exception as e:` debug log, then FALLS THROUGH to the live fallback (1798–1799).
  - Live fallback:
    - try 1803; per term `snippet = await self._get_wiki_snippet_cached(term)` (1818), which swallows (returns None, 559–599): a read-only sibling;
    - `except Exception as e:` warning → `return []` (1824–1826): ANCHOR #80;
    - `finally` records `fallback_ms`.
- `_get_semantic_chunks` (wrapper, def ≈1830) is try/finally only. `_get_semantic_chunks_timed` (def 1841):
  - `if not query: return []` (1843–1844).
  - in-flight guard `if not _WIKI_SEM_INFLIGHT.acquire(blocking=False): … return []` (1850–1855).
  - `_search_and_release` calls `semantic_search_with_neighbors(query, k)` (1859). Since F4 it returns an OutcomeList (no_results, unavailable "index_not_loaded", failed <class>, succeeded).
  - try 1866: the awaited search; `if not results: return []` (1881, a falsy check only); then `results = [r for r in results if r.get("similarity", 0) >= SEMANTIC_CHUNKS_GATE_THRESHOLD]` (1886–1887, drops status); …; `chunks = list(chunks_by_title.values())` (1945); `return chunks[:max_results]` (1946).
  - `except asyncio.TimeoutError:` sets `timings["timed_out"] = True`, warns (1948–1950), falls to the tail.
  - `except Exception as e:` warns (1951–1952), falls to the tail.
  - trailing `return []` (1954). Not a packet anchor; same class.
- knowledge/semantic_search.py `SemanticSearchIndex.search` (def 321):
  - step 4 `row_data_map = self._read_rows([i for i, _ in hits])` (363, outside any try);
  - step 5: for each hit, `if not data: continue` (370); `return OutcomeList(rows[:k])` (377).
  - `_read_rows` returns `{}` when `not self._pq_file or not indices`. It catches each row-group read error, logs a warning and continues. Hits are pre-filtered to `idx < self._total_rows` (356).
  - The `search` docstring wrongly says a not-loaded index is no_results; it is unavailable (F4 parent review).
- Existing tests that touch these paths (parent/agent grep; re-grep):
  - tests/unit/test_sep09_latency_metrics.py (parametrized task="wiki"/"semantic" timeouts assert `== []`; `test_wiki_records_swallowed_live_snippet_timeout`);
  - tests/unit/test_audit0831_fixes.py (TestWikiTimeoutSkip);
  - tests/unit/test_semantic_visual_failure_outcomes.py;
  - tests/unit/test_semantic_search_metric.py;
  - tests/unit/test_semantic_load_concurrency.py;
  - tests/unit/test_doc_cooccurrence.py;
  - tests/unit/test_hybrid_semantic_score.py.

OWNERSHIP
- core/prompt/gatherer_knowledge.py: the bodies of `_get_wiki_content_timed` and `_get_semantic_chunks_timed` ONLY (plus the import, only if F7a did not add it).
- knowledge/semantic_search.py: `SemanticSearchIndex.search` ONLY (body and docstring). `_read_rows` stays unchanged.
- New tests/unit/test_gatherer_outcomes_background_knowledge.py.
- New docs/execution/generalization/batches/F7b.md.
- The response file named above.
- Read-only: every other method in both files, `_get_wiki_snippet_cached`, knowledge/doc_cooccurrence.py, builder, formatter, orchestrator, handlers, utils/retrieval_outcome.py, config/**, docs/execution/generalization/briefs/**, and everything else. Every class-guard-owned path is also read-only (see R_common_rules.md "Never edit").

CONTRACT
1. Wiki (`_get_wiki_content_timed`):
   - in-flight guard → `OutcomeList.unavailable("in_flight")`;
   - chroma TimeoutError → `OutcomeList.unavailable("timeout")`, with `timings["timed_out"]` and the warning unchanged;
   - chroma other exception: keep the fall-through to the live fallback, and remember `chroma_err = type(e).__name__`;
   - fallback outer except (ANCHOR #80) → `OutcomeList.failed(type(e).__name__)`;
   - chroma raised AND the fallback returned no results → `OutcomeList.failed("chroma:" + chroma_err)`;
   - chroma raised AND the fallback returned results → succeeded with those results. The fallback is the designed replacement; record this rationale;
   - empty query and `_should_skip_wikipedia` stay `[]` (no_results).
2. Semantic (`_get_semantic_chunks_timed`):
   - in-flight guard → `OutcomeList.unavailable("in_flight")`;
   - immediately after the awaited search and BEFORE the 1881 falsy check: `sem_status, sem_reason = outcome_status(results)`; if failed/unavailable → return an `OutcomeList` with that status and reason (no items);
   - TimeoutError → `OutcomeList.unavailable("timeout")` (timings unchanged);
   - other exception → `OutcomeList.failed(type(e).__name__)`;
   - healthy paths return exactly today's chunks.
3. `SemanticSearchIndex.search` total row-read failure. After `row_data_map = self._read_rows(...)`, when `hits` is non-empty and `row_data_map` is empty:
   - `return OutcomeList.failed("row_read_failed")` if `self._pq_file` is set;
   - else `return OutcomeList.unavailable("metadata_unavailable")`.
   - A partial read (some rows read) stays succeeded with the rows read. Record why: a single bad row group must not turn every hit query into a failure, and there is no "partial" state.
   - Fix the docstring so it lists the states exactly as implemented.
4. No change to thresholds, dedupe, stitching, timings keys, in-flight semaphore handling, log text or fallback behaviour beyond the returns above.
5. Privacy: reasons are constant labels or exception class names only.

TESTS (tests/unit/test_gatherer_outcomes_background_knowledge.py; fakes only; NO REAL INDEX)
- Reuse the fakes and techniques of test_sep09_latency_metrics.py and test_audit0831_fixes.py: blocked queries plus a monkeypatched timeout constant; holding `_WIKI_CHROMA_INFLIGHT` / `_WIKI_SEM_INFLIGHT` to simulate busy.
- FAILING FIRST in one command: `sha256sum core/prompt/gatherer_knowledge.py knowledge/semantic_search.py`, then the new tests on the UNEDITED sources. List the failures.
- Wiki, through the deployed `_get_wiki_content`:
  - busy → unavailable / in_flight;
  - timeout → unavailable / timeout, and `timed_out` is still recorded;
  - fallback raising → failed / class;
  - chroma raising with an empty fallback → failed / "chroma:<class>";
  - chroma raising with a fallback that has items → succeeded with those items (control);
  - healthy chroma → today's results.
- Semantic, through the deployed `_get_semantic_chunks`:
  - busy → unavailable / in_flight;
  - producer `OutcomeList.failed("X")` → failed / X;
  - producer `OutcomeList.unavailable("index_not_loaded")` → unavailable;
  - timeout → unavailable / timeout;
  - an unexpected exception inside the pipeline → failed / class;
  - healthy → today's chunks;
  - healthy but everything below the threshold → no_results.
- `SemanticSearchIndex.search` (fake index, fake encoder, fake `_pq_file` whose `read_row_group` raises; never `get_index()`):
  - all rows unreadable → failed / row_read_failed;
  - `_pq_file` None with hits → unavailable / metadata_unavailable;
  - one of two row groups unreadable → succeeded with the readable rows (control);
  - end to end: monkeypatch `knowledge.semantic_search.get_index` to that fake index → `knowledge.doc_cooccurrence.doc_cooccurrence` raises `RetrievalError` (reason semantic_failed) instead of returning known=False.
- Through the builder (F5 integrated), if the `full_builder` fixture creates the task: a busy semantic slot → `_section_outcomes["semantic"] == {"status": "unavailable", "reason": "in_flight"}`.
- Privacy: a distinctive marker in the query and the exception messages never appears in any reason.
- Focused: the new file plus the 7 existing files above (one chunk of 8).
- Sweep:
  - every remaining tests/unit importer of core.prompt.gatherer_knowledge or knowledge.semantic_search (grep and list), in chunks of ≤9.
  - Non-unit importers only under NON-UNIT TESTS rules.
  - Never run tests/test_web_search_manager.py or tests/test_prompt_internal_methods.py.

SCAN: pre- and post-edit read-only scan. Expect #80 STALE and new = 0. Also report whether the 1951/1954 semantic site was a dm18 finding before; if it was, and it is not in the baseline, record it.
RESPONSE FILE:
- Include the R_common_rules contents for #80.
- Record the semantic-chunks sibling and the `SemanticSearchIndex.search` row-read sibling, each with deployed-function evidence.
- Record the `_get_wiki_snippet_cached` swallow as a read-only sibling.
SIZE: target ≤380 changed lines; HARD stop at ≤450. Stop before exceeding it and return a split proposal; the natural split is wiki vs semantic plus row reads.
ORDER: manifest → create F7b.md → pre-edit scan → tests → failing-first (digests in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
