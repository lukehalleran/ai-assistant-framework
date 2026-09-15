=== F8 parent fact notes (collected 2026-09-14; re-verify at F8 Ready) ===
(Durable copy, re-created in the repo on 2026-09-14 after a machine crash wiped the /tmp scratchpad. The F8 brief itself is not written yet; it waits for the F7 results.)

Scope per failure_outcome_design.md: F8 = memory gatherer (gm) + web gatherer (gw #92 evidence). Request: CGR-20260913-007 (packet sha b3f24947…). F8's response will be CGR-20260913-007-4.md.

DIGESTS (post-F4 tree; all equal the packet's source SHA-256)
- core/prompt/gatherer_memory.py 2cbe7775…
- core/prompt/gatherer_web.py 270b6208…
- core/prompt/gatherer_knowledge.py 77d094d6… (F7a–F7c change this one)

ON-PATH gm ANCHORS → builder section (builder lines are pre-F5; F5 shifted builder lines by about +7 before the gather loop and about +22 after it)
- #87 `MemoryRetrievalMixin._get_recent_conversations` (def 169; except 227–229 → `return []`) → task "recent" (builder 1261).
  - Direct callers OUTSIDE the gather loop, each already inside its own try:
    - builder 1816: Step 6.1 memory top-up, try 1799 / except 1839;
    - builder 1947: inner try 1946 / except 1948, inside an outer try 1940;
    - builder 2291: `_build_lightweight_context` (def 2286), try 2289. Locate its except at Ready.
  - If F8 makes this method raise, verify each direct caller keeps its current fallback, and state it. The F7 decision is a typed return (`OutcomeList.failed(...)`), which keeps direct callers unchanged; F8 should mirror it.
- #88 `_get_semantic_memories` (def 489; except ≈630–634 with a traceback debug → `return []`) → "memories" (1267).
- #90 `get_user_profile_context` (def 927; except 962–964 → `return ""`) → "user_profile" (1274). The value is a str.
- #91 `get_upcoming_schedule` (def 966; except 1068–1070 → `return []`) → "upcoming_schedule" (1443). Its success return `upcoming[:min(limit, SCHEDULE_PROMPT_MAX_EVENTS)]` is a slice, which drops any status.

OFF-PATH gm ANCHORS (design: evidence, not code)
- #85 `get_recent_facts` (def 144; except 153–155). Falls back to `self.get_facts(limit)` (152) and calls `memory_coordinator.get_recent_facts` (148).
- #86 `get_facts` (def 157; except 165–167). Calls `memory_coordinator.get_facts` (161).
- #89 `_get_reflections` (def 755; except 797–799). The builder uses `_get_reflections_separate` (1300) instead.
- No production caller of these MIXIN methods (parent grep of api core gui memory utils knowledge scripts integrations):
  - core/context_pipeline.py:973 `self.memory_system.get_facts`: memory_system is the memory coordinator (docstring 283);
  - memory/shutdown_processor.py:1313 `mc.get_facts`: the coordinator;
  - scripts/sample_real_benchmark.py:552 `retriever.get_facts`: the retriever;
  - memory/memory_coordinator.py:328/335: the retriever.
  - Evidence test idea: prove the builder never calls them (monkeypatch them to raise, then build a prompt through `full_builder`).

gw #92
- `WebSearchMixin._get_web_search_results` (def 68; except 301–304): sets `self.last_web_decision["error"] = type(e).__name__`, warns, `return None`.
- Existing receipt chain: `web_search_decision` context key → formatter `web_search=ON(error)` (F6a left it unchanged) → turn record `web_error`.
- GAP after F5: `outcome_status(None)` gives ("no_results", ""). `_section_outcomes["web_search"]` would read no_results while the web receipt says error, and F6b's `sections_not_checked` omits web_search.
  - F8 decides how to make them agree: raise after setting the receipt (F5 records failed/<class>; gathered becomes []), or a typed return.
  - First check the builder's consumers of `gathered["web_search"]` (None vs [] vs dict handling).
- F2 follow-up in the same area: `WebSearchResult.extract_error` surfacing (named in F6b's out-of-scope list).

DICT-SECTION TRAP IN gm (from F5.md §14)
- `summaries` ← `_get_summaries_separate` (gatherer_memory.py:312) and `reflections` ← `_get_reflections_separate` (gatherer_memory.py:819) both return `{"recent": [...], "semantic": [...]}`.
- That dict is ALWAYS truthy, so F5 records "succeeded" even when both lists are empty or a read failed.
- Neither is a CGR-007 anchor, but both are the same class, as is `visual_memories` (gatherer_knowledge; F7 left it out).
- Decide at F8 Ready: typed inner lists, or a documented limitation. Changing the shape would need a builder change, and F5 owns the builder.
- `user_profile` returns str: truthiness already matches content, so it is not a trap.

BUILDER WRAPPERS (verified; they do not swallow gatherer execution)
- google_calendar / relevant_emails / daemon_self_notes task creation sits in `try: … except Exception: pass`. That guards only the config import and `asyncio.create_task`; the gatherer coroutine runs later in the gather loop.
- proactive_insights cold cache: a background `_warmup_insights` with its own except, and NO task. The section is absent, meaning "not attempted".
- gatherer_knowledge `_get_wiki_content` and `_get_semantic_chunks` wrappers are try/finally only (no second swallow).

TO CONFIRM AT F8 READY
- F5.md (dict-section list, exact recording), F6a.md, F6b.md parent sections.
- F7a/F7b/F7c results (the typed return per site) so F8 mirrors them.
