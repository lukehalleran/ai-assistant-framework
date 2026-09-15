=== F8a: memory gatherer sections report a typed failure instead of an empty section (recent conversations, semantic memories, user profile, upcoming schedule; off-path evidence for facts/reflections) ===
(Parent draft, 2026-09-14. It moves to docs/execution/generalization/briefs/F8a.md after F7c is integrated. The design-doc F8 split and limitations amendment is written at the same time.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Decisions per request" → CGR-007, including "Off-path sites #85, #86 and #89 are answered as not on the prompt path; they get evidence, not code".
- "F7 split and gatherer outcome shape (parent decision, 2026-09-14)": typed return, not raise; a str-returning gatherer re-raises.
- The parent's F8 split (to be recorded in the design doc at launch): F8a = memory gatherer, response `CGR-20260913-007-4.md`; F8b = web gatherer #92, response `-5`.
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md. This batch answers anchors #85–#91: #87, #88, #90 and #91 with code; #85, #86 and #89 with off-path evidence.
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-007-4.md.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL: "Response file", SHELL GUARD, NON-PYTEST CODE, GIT INDEX AND PYTEST HYGIENE, NON-UNIT TESTS (including the tests/unit/test_graph_integration.py exclusion and the ContextGatherer/UserProfile pattern), INTERRUPTION and MEMORY.

COMPLIANCE (read twice; F7a and F7b broke these):
- Your FIRST source-file write comes AFTER the failing-first run is recorded in batches/F8a.md.
- Run NO non-pytest Python (no `python -c`, `python3 -`, heredoc, REPL or script). The exceptions are `python -m ruff`, the read-only scan and the single `import utils` sanity check. Use grep/jq for scan JSON.
- Check each fixture input against the gates it must pass. A control that fails on the unedited source means your fixture is wrong.
- Disclose any breach in the packet immediately.

MANIFEST CHECK (before any edit; if either check fails, stop and report)
Run from the checkout root with S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:
  (a) `sha256sum -c --quiet $S/manifest_post_F7c.txt` must print nothing and exit 0.
  (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F7c.txt` must print nothing.
GUARD, MEMORY, DATA NOTE: as in R_common_rules.
- Record the data/ baseline before the first pytest: 7 entries, including user_profile.json (598 bytes) which must stay untouched. logs/ is absent.
- Re-check after every chunk; STOP on any change.
ORDER REMINDER: print `sha256sum core/prompt/gatherer_memory.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old flattening may be repaired only with the new assertion plus a paired control. List every edit.

PARENT-VERIFIED FACTS (read-only mapping of core/prompt/gatherer_memory.py `2cbe7775…` = the packet's source sha, 1070 lines; re-verify at the manifest check)
- #85 `get_recent_facts` (def 144):
  - try 146–154; `except Exception` 153 → `return []` 154–155;
  - producer `await self.memory_coordinator.get_recent_facts(limit)` (148; raises);
  - capability fallback `return await self.get_facts(limit)` (152).
- #86 `get_facts` (def 157):
  - try 159–166; except 165 → `return []` 167;
  - producer `memory_coordinator.get_facts` (161); capability gate 163–164.
- #89 `_get_reflections` (def 755): except 797–799 → `return []`. The builder uses `_get_reflections_separate` instead.
- OFF-PATH (parent-verified by grep of core, gui, api, scripts, memory, utils, knowledge): no production caller invokes the MIXIN's `get_recent_facts`, `get_facts` or `_get_reflections`.
  - core/context_pipeline.py:973 and memory/shutdown_processor.py:1313 call the memory COORDINATOR's `get_facts`.
  - scripts/sample_real_benchmark*.py call a retriever's.
  - memory/memory_coordinator.py:328/335 delegate to the retriever.
- #87 `_get_recent_conversations` (def 169; builder task "recent" at builder.py 1268):
  - `corpus_manager.get_recent_memories(count=limit)` (177) is unguarded, so an exception reaches the anchor except;
  - an inner try 183–194 (`except` 193) swallows a FALLBACK-only failure and continues with the existing memories. That is a sibling; leave it and record it;
  - annotation comprehension 199; `self.memory_id_map[...]` 217;
  - ANCHOR except 227–229 → `return []`.
  - Direct callers OUTSIDE the gather loop (builder.py):
    - 1838 (Step 6.1 top-up, try 1821–1862);
    - 1969 (recency floor, local try 1968–1972);
    - 2314 (`_build_lightweight_context`, def 2309, try 2312–2374, whose except returns a DIFFERENT fallback dict).
  - A typed `OutcomeList.failed(...)` return (equal to [] and falsy) keeps all three unchanged. A RAISE would change 2314's behaviour, so #87 MUST return typed, never raise.
- #88 `_get_semantic_memories` (def 489; builder task "memories" at 1274):
  - `if not query: return []` (502–504) is a legit gate;
  - `semantic_memories = []` (510); inner try 511–594 around `await self.memory_coordinator.get_memories(...)` (522/528) and its processing. Inner `except Exception` 593–594 SWALLOWS a coordinator failure and continues with an empty list. This is the class instance on the real failure path;
  - transforms: `_apply_valence_cap` 590, `_deduplicate_memories` 597, slice `result[:limit]` 598, annotation 606; `memory_id_map` 617;
  - ANCHOR outer except 630–634 (with a traceback debug) → `return []`.
- #90 `get_user_profile_context` (def 927; builder task "user_profile" at 1281; its ONLY caller):
  - returns str on every path; `if not self.user_profile: return ""` (939–941) is a legit gate;
  - producer `self.user_profile.get_context_injection(...)` (944, sync) sits in try 943–964;
  - `memory_id_map["PROFILE_CONTEXT"]` 952–959;
  - ANCHOR except 962–964 → `return ""`.
  - Consumers already handle a non-str value: the gather loop stores `raw or []`; builder.py:800 skips user_profile in the oversize scan; formatter.py:1708–1709 renders only `if user_profile and isinstance(user_profile, str)`.
- #91 `get_upcoming_schedule` (def 966; builder task "upcoming_schedule" at 1450; its ONLY caller):
  - legit early returns: SCHEDULE_EXTRACTION_ENABLED off (983–984), no chroma store (988–989), empty collection (993–994), no schedule facts (1012–1013);
  - producer `store.query_collection("facts", ...)` (998) raises;
  - success returns the slice `upcoming[:min(limit, SCHEDULE_PROMPT_MAX_EVENTS)]` (1066);
  - ANCHOR except 1068–1070 → `return []`.
- NOT in scope (parent decision, recorded as a limitation): `_get_summaries_separate` (312, except 386–388) and `_get_reflections_separate` (819, except 893–895) always return a truthy `{"recent": [...], "semantic": [...]}` dict, so F5 records them as succeeded even on failure. The feature inventory never itemizes them. Fixing that needs a builder change; leave both untouched and record them.
- Existing tests: no unit test drives these methods' own except blocks. Builder-level mocks exist in tests/unit/test_section_outcomes.py and tests/unit/test_prompt_timeout.py; happy-path mixin use exists in tests/unit/test_sep10_probe_dump_interpretation.py (1652–1810). tests/unit/test_retrieval_pool_caps.py, test_keyword_anchor_retrieval.py, test_health_transient_retrieval.py and test_recency_metadata_fallback.py use MemoryRetriever methods with the same names, NOT the mixin.
- Mirror F7c's integrated narrative approach for #90 (a str gatherer re-raises). Re-read batches/F7c.md, including its parent section, and confirm.

OWNERSHIP
- core/prompt/gatherer_memory.py, ONLY:
  - one import line (`from utils.retrieval_outcome import OutcomeList`);
  - the bodies of `_get_recent_conversations`, `_get_semantic_memories`, `get_user_profile_context` and `get_upcoming_schedule`.
- `get_recent_facts`, `get_facts` and `_get_reflections` stay UNCHANGED (off-path evidence only).
- New tests/unit/test_gatherer_outcomes_memory.py.
- New docs/execution/generalization/batches/F8a.md.
- The response file named above.
- Read-only: every other method in the file (including `_get_summaries_separate` / `_get_reflections_separate`), core/prompt/gatherer_web.py, gatherer_knowledge.py, builder, formatter, memory/**, utils/retrieval_outcome.py, config/**, briefs/**, LANDING_NOTES.md, and every class-guard path.

CONTRACT
1. #87: the anchor except returns `OutcomeList.failed(type(e).__name__)`, with the log line unchanged. The fallback-only inner swallow (193) is unchanged and recorded as a sibling.
2. #88:
   - the inner except (593) keeps its log and records `retrieval_err = type(e).__name__` (initialise it to None before the inner try);
   - at the success return, if `retrieval_err` is set, return `OutcomeList(<final list>, status="failed", reason=f"retrieval:{retrieval_err}")`, otherwise the final list as today;
   - the outer anchor except returns `OutcomeList.failed(type(e).__name__)`;
   - `if not query` stays `[]`.
3. #90: the anchor except keeps its warning log and then re-raises (`raise`). The gather loop records failed/<class>, and the gathered value becomes `[]`, which consumers already treat as no profile. The `not self.user_profile` gate still returns "".
4. #91: the anchor except returns `OutcomeList.failed(type(e).__name__)`. Every legit early return stays `[]`.
5. #85, #86 and #89: NO code change. Prove they are off-path with a deployed builder test: monkeypatch the three mixin methods to raise AssertionError, build a prompt through `full_builder` with facts/reflections limits enabled, and assert the build succeeds and none was called. Record the grep evidence (above) in the packet and the response.
6. No change to transforms, dedup, caps, annotation, memory_id_map, log text or any other method.
7. Privacy: reasons are constant labels or exception class names only.

TESTS (tests/unit/test_gatherer_outcomes_memory.py; fakes only; no real Chroma, corpus store, UserProfile file or model)
- Build the mixin the way the existing happy-path tests do (see test_sep10_probe_dump_interpretation.py's mixin setup). Give any coordinator a `user_profile` attribute or a fake; never construct `UserProfile()` on the default path.
- FAILING FIRST in one command: `sha256sum core/prompt/gatherer_memory.py`, then the new tests on the UNEDITED source. Record the failures.
- #87: a raising `corpus_manager.get_recent_memories` → failed / class and `== []`; healthy non-empty → today's items; healthy empty → no_results.
- #88:
  - a raising coordinator `get_memories` (inner swallow) → failed / "retrieval:<class>" and `== []`;
  - an exception outside the inner try (e.g. a raising `_deduplicate_memories`) → failed / class;
  - empty query → `[]` / no_results; healthy → today's items.
- #90: a raising `get_context_injection` → the method raises (pytest.raises); through `full_builder`, `_section_outcomes["user_profile"] == failed/<class>` and the prompt still builds; no profile → "".
- #91: a raising `query_collection` → failed / class; disabled flag → `[]` / no_results; healthy → today's slice.
- #87 direct caller: through `_build_lightweight_context` (the light path), a raising recent-memories store still returns the NORMAL light context (not the except fallback dict). This proves the typed return keeps that caller unchanged.
- Off-path evidence test (contract point 5).
- Privacy: a distinctive marker in the query and the exception messages never appears in any reason.
- Focused: the new file plus tests/unit/test_section_outcomes.py, test_prompt_timeout.py, test_sep10_probe_dump_interpretation.py, test_independent_prompt_audit.py and test_light_prompt_path.py (≤9 per chunk).
- Sweep:
  - every remaining tests/unit importer of core.prompt.gatherer_memory / MemoryRetrievalMixin / core.prompt.context_gatherer (grep and list), in chunks of ≤9, excluding test_graph_integration.py and any file matching the ContextGatherer/UserProfile hazard;
  - never run tests/test_web_search_manager.py or tests/test_prompt_internal_methods.py.

SCAN: pre- and post-edit read-only scan. Expect the dm18 rows for #87, #88, #90 and #91 to go STALE and #85, #86 and #89 to remain LIVE (evidence-only, by design), with new = 0. A re-raising except is not a dm18 finding; confirm it.
RESPONSE FILE (`-4`):
- #87, #88, #90, #91 fixed, with deployed-function evidence;
- #85, #86, #89 answered as NOT on the prompt path, with grep and builder-test evidence; the class-guard owner decides detector vs accepted debt;
- the siblings: #87's fallback-only inner swallow, and the summaries/reflections dict trap as a limitation;
- #92 → F8b (`-5`).
SIZE: target ≤380 changed lines; HARD stop at ≤450.
ORDER: manifest → create F8a.md → pre-edit scan → tests → failing-first (digest in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
