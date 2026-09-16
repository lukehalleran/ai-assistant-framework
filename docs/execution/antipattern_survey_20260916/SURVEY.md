# Daemon_v1 anti-pattern survey (read-only, 2026-09-15)

File universe (`git ls-files '*.py'`, excl. data/venv/node_modules/integration.bak): 1017 files — app=313
(`core memory knowledge utils gui api models processing config agent_branch eval` + `main.py`), tests=553,
scripts=131, other/misc=20 (not analyzed as a group). Counts via one AST pass (`analyze.py`, `ast.dump`-based
dedup) run as `env -u PYTHONPATH DISABLE_FS_GUARD=1 PYTHONDONTWRITEBYTECODE=1 python -s analyze.py`, plus
targeted `rg` cross-checks shown inline. Full per-file detail lives in `antipattern_counts.json`.

## 1. Function-body imports
| category | app | tests | scripts |
|---|---|---|---|
| stdlib | 159 | 202 | 21 |
| project module | 698 | 1772 | 205 |
| config.app_config | 306 | 94 | 38 |
| heavy third-party | 45 | 10 | 14 |
| other | 61 | 28 | 12 |
| **total** | **1269** | 2106 | 290 |
| carry `# lazy import:` marker | 57 | 0 | 1 |

Module-level `from config.app_config import X`: 37 statements. `config.app_config` function-body imports
concentrate in `core/prompt/*` — the project's own documented exception (import doctrine case 3, live-config
read). Only 57/306 carry the `# lazy import:` marker prescribed for the other three cases, so a case-3 read
is indistinguishable from an un-reviewed one. Closure: doctrine comment for case-3 sites. Effort S.

## 2. `except Exception` / bare `except:`
| | app | tests | scripts |
|---|---|---|---|
| total | 1189 | 47 | 128 |
| swallow-only (pass/continue/return/`.debug`) | 412 | 11 | 67 |
| other handling | 777 | 36 | 61 |

Top-5 app files by total: `gui/handlers.py`(88), `core/orchestrator.py`(46), `core/prompt/builder.py`(36),
`memory/shutdown_processor.py`(33), `core/prompt/gatherer_knowledge.py`(32). By swallow-only:
`gui/handlers.py`(28), `core/orchestrator.py`(18), `memory/shutdown_processor.py`(17), `core/agentic/gate.py`(14).
Some IS the architecture ("Graceful degradation" doctrine) but 412 silent swallows is ~10x the ~3 documented
degradation paths, and this mechanism underlies BC-69 (silent ops failures — Backblaze dead 46 days,
daily-notes timer dead 7.5 months) and BC-70 (log severity misdescribes control flow). Closure: every
swallow-shaped handler names its degradation path in a comment, or narrows to a specific exception; AST lint
buildable. Effort M (triage), S (lint alone).

## 3. Try blocks per file
Total: app=1831, tests=207, scripts=220. Top-5 app: `gui/handlers.py`(108), `memory/shutdown_processor.py`(69),
`core/orchestrator.py`/`gui/launch.py`(55), `core/prompt/builder.py`(49). Functions >50% try-body: app=67,
tests=21, scripts=26. Correlates 1:1 with §2's file list — same habit of wrapping single risky calls in
try/except per-call rather than centralizing the risk. Closure: same as §2. Effort M.

## 4. `config.app_config` import style
Module-level: 37 statements / 57 distinct files. Function-body (live read): 306 statements / 176 distinct
files — 8x more files use the live-read form. Mostly healthy per BC-11 ("live setting never reaches an
already-built consumer" — frozen imports are the textbook trigger, e.g. 2026-09-09 `gatherer_web`). The 37
module-level sites are the residual BC-11 risk surface. Closure: review each of the 37 for a Settings-page
toggle; extend `test_sep09_live_controls.py`'s same-instance probe. Effort M.

## 5. `print(` calls in app code
Raw: app=525, tests=731, scripts=1923. Excluding `gui/wizard.py`+`main.py` (documented CLI): app=260 —
`gui/launch.py`(69, startup banners), `eval/run_phase{2,4,5,6}.py`(69 combined, CLI-runner-shaped),
`utils/bootstrap.py`(24), `utils/startup.py`(21), `utils/keyword_matcher.py`(11, no CLI justification).
Bypasses log-level control, `DAEMON_TEST_MODE` isolation, log rotation. Closure: ruff `T201` scoped to app
dirs minus the 2 documented exceptions. Effort S.

## 6. `global` statements / module-level mutable singletons
`global`: app=60. Distinct (file,name) pairs: 62; 27 refs (25 distinct names) target a module-level
`NAME = None` — the lazy-singleton pattern. Top: `main.py`(4), `knowledge/semantic_search.py`/
`core/actions/google_contacts.py`(3). Most are the documented case-1 lazy singleton (CLIP/SentenceTransformer/
spaCy warmup, correct per doctrine); a minority are ad hoc global mutable state with only a per-store test
fixture for isolation. Closure: guard test enumerating `global`-touched names, asserting each has a reset
fixture. Effort S/M.

## 7. Mutable default arguments
**Zero** instances (`def f(x=[]|{}|set())`) anywhere — verified by AST pass and an independent `rg`. Clean;
listed for completeness only.

## 8. `sys.path.insert/append` outside `scripts/`/`tests/`
1 app instance: `main.py:113` (PyInstaller bootstrap, documented). scripts=107, tests=28 (expected path
shims). Not a problem in app code.

## 9. `os.environ`/`os.getenv` outside `config/`
**252** calls, **186 distinct names**, outside `config/`. Top: `memory/shutdown_processor.py`(22),
`knowledge/WikiManager.py`(16), `utils/query_checker.py`(13), `core/prompt/gatherer_knowledge.py`(10).
CLAUDE.md states the config pipeline is YAML→schema→app_config-with-env-override; these 186 names bypass it
entirely (no schema validation, no `config.local.yaml` path, no live-Settings reach). Sibling of BC-12 in the
opposite direction — no config key exists at all. Closure: chokepoint — route through `app_config` or an
audited `utils.env.get()` wrapper. Effort L (volume), S (wrapper).

## 10. Naive datetimes
Naive `datetime.now()`/`.utcnow()`: app=176, tests=285, scripts=89 (550 total). tz-aware: app=30, tests=16,
scripts=1 (≈6:1 naive:aware in app). Two purpose-built tz modules (`utils/timezone_resolver.py`,
`utils/temporal_resolver.py`) exist because of live incidents (Sep-4 deadline bug, Sep-5 pacing bug); 550
naive sites elsewhere are the same latent shape. Closest catalog entry: BC-58 (per-incident, not a rule).
Closure: prefer the tz helpers project-wide; see candidate class below. Effort L.

## 11. Blocking calls inside `async def`
| kind | app | tests | scripts |
|---|---|---|---|
| `time.sleep(` | 0 | 3 | 0 |
| `requests.*` | 0 | 0 | 0 |
| `subprocess.run(` | 8 | 0 | 0 |
| `open(` non-tmp | 5 | 4 | 33 |
| `.encode(` (embedder proxy) | 15 | 2 | 2 |

`subprocess.run` in async: `agent_branch/goal_runner.py`(3), `gatherer_knowledge.py`(4, git-log). `.encode(`
in async (15 app, name-based proxy) risks the same CPU-encode-blocks-the-loop class already fixed once for
the memory gate's GPU autodetect. Closure: `asyncio.to_thread` (already used for `ingest_image`). Effort S/M.

## 12. Deprecated asyncio APIs
`asyncio.get_event_loop()`: app=11 (deprecated 3.10+). `ensure_future` app=11 vs `create_task` app=34.
Closure: mechanical replacement; no built-in ruff rule, repo-local AST check buildable. Effort S.

## 13. `hasattr`/`getattr` lazy-attribute idiom
`hasattr`: app=257. `getattr`: app=705. `if not hasattr(self, "_x")` lazy-init: **103** app instances
(`core/agentic/controller.py`/`protocols.py`, `core/context_pipeline.py`, `core/orchestrator.py`,
`core/insight/*`). Untyped alternative to `__init__` defaults/`cached_property`; tolerates a typo'd attribute
name silently. Closure: prefer explicit fields/`cached_property`; doctrine note. Effort M.

## 14. `getattr(app_config, "NAME", default)`
Only **4** instances, all `knowledge/implementation_detector.py`. Below the 10-instance floor — not
widespread, dropped from ranking.

## 15. Duplicate functions / same-name helpers
Identical-AST-body groups (≥2 members, ≥4 stmts): **34**. Notable: `core/prompt/base.py`/`formatter.py:
_dedupe_keep_order`; two IN-FILE duplicates — `web_search_manager.py:get_formatted_content` on both
`WebSearchResult`(278) and `MultiSearchResult`(373), `memory_retriever.py:get_item_id` nested twice (921,
1057), neither propagates a fix to its sibling; `completed_plan_claims.py`/`streak_claims.py:_coerce_date`;
`daily_notes_generator.py`/`monthly_notes_generator.py:_get_week_folder_name`; `agent_branch/workers/*.py:
connect` (3 files). Same-name helper in ≥3 modules: 276 (mostly generic OOP noise). Targeted `rg` recount:
`_slugify`=4, `_coerce_date`=3, `_truncate`=7, `_dedupe_keep_order`=3, `_daemon_running`=**19** (17 in
`scripts/`, despite the 2026-08-21 centralized `utils/daemon_guard.py` built for exactly this — BC-58
sibling). Closure: hoist shared helpers; finish the `_daemon_running` migration. Effort S/helper, M aggregate.

## 16. Size
Files >2000 lines: **13** app (`gui/handlers.py` 6273, `core/agentic/controller.py` 3748,
`core/orchestrator.py` 2682, `knowledge/web_search_manager.py` 2584, `core/prompt/builder.py` 2535, +8 more)
+ 1 test file. Functions >150 lines: **80** app (top: `controller.py:run_agentic_search` 1358,
`builder.py:build_prompt` 1234, `formatter.py:_assemble_prompt` 1071). Classes >1000 lines: **13**
(`AgenticSearchController` 3427, `DaemonOrchestrator` 2221, `ToolExecutor` 2101, `ShutdownProcessor` 2003,
`UnifiedPromptBuilder` 1797, +8 more). Same ~10 files dominate §2/§3 too. Closure: refactor backlog, out of
scope for a quick fix. Effort L.

## 17. Lint suppressions
`# type: ignore`: app=12. `# noqa`: app=28. `# pragma: no cover`: app=6. Low volume, consistent with the
project's stated "don't add style rules casually" stance (few suppressions because few rules enforced yet).
## 18. TODO/FIXME/HACK/XXX
Real comment markers (excl. string-literal data like placeholder-detector tuples): app=6, tests=4,
scripts=17; none dated. Below the 10-instance floor. Positive finding: incident fixes go into
`CLAUDE_CHANGELOG.md`/`FOLLOWUPS.md` instead of accumulating as TODOs.

## 19. `re.compile(` location
Module level: app=623. In-function: app=27. ~96% module-level — healthy, not a problem.

## 20. Direct store I/O bypassing `utils/safe_json.py`
AST-detected `open(path,"w")` on a literal `.json`/`data/` path: **0** in app. All `open(...,"w")` in app
(excl. `safe_json.py`): only **12**, all either `safe_json.py` itself, `backup_manager.py`/`fs_snapshot.py`,
or non-JSON files (`.env`, `config.local.yaml`, hash cache, devnull, log redirect). `json.dump(` in app:
only **3** sites, same 3 files. **Doctrine is well-enforced** — clean result despite CLAUDE.md's emphatic
Critical Rule (the 2026-07-14 entity_aliases/claim_index incident). Closure: cheap guard test to keep it
that way. Effort S.

## 21. Enum-to-string comparisons
Raw regex hits: app=524, tests=445, scripts=77 — but a 25-line manual triage found ~90%+ false positives
(`str(Path)`, `str(exception)`, correct `str(x.value)`). Precise re-check: the dangerous shape
(`"lit" in str(<obj>)` or `=="Enum.MEMBER"`) has **0** live enum instances in app and only 9 residual
non-enum `in str(` sites. Matches BC-01's flagship incident (`"crisis" in str(CrisisLevel...)`, fixed
2026-08-27/09-12) — **class closed for the enum case today**; raw count is a measurement artifact. No action
beyond keeping the existing `trigger_match.py`/`dm01_raw_substring` guard.

## 22. Tests
Date-named files (`test_sep\d\d_*`/`test_\d{8}_*`/`test_audit\d+_*`): **53**/553 (9.6%), **1168** `def test_`
functions (top: `test_sep10_probe_dump_actions.py` 168, `test_sep10_probe_dump_interpretation.py` 123).
Files >1500 lines: 3. `time.sleep(` in tests: 15 files. Tests reading source/`inspect.getsource` as a
contract guard: **88 files** — checks source strings rather than behavior, BC-63's risk shape when that's
the ONLY check. Monkeypatch `sys.modules`: 8 files. Dated-file naming is explicit project policy (each
incident batch gets its own regression file) — intentional, but fragments a subject's coverage across many
files. Closure: doctrine tradeoff; periodic subject-consolidation pass. Effort L, not urgent.

## 23. Logging style
Eager f-string/pre-formatted logger calls: app=**2031** vs lazy/other-arg app=555 (≈3.7:1). Acquisition:
custom `get_logger(...)` 147 app sites vs raw `logging.getLogger(__name__)` 50 — two competing conventions,
worth checking the 50 get the same handler config. Closure: ruff `G004`/`G002`; doctrine note on canonical
acquisition style. Effort S.

## 24. Dated comments embedded in source
Lines starting `# YYYY-MM-DD`: app=98, tests=35, scripts=6. Top: `core/agentic/gate.py`(12),
`core/action_claim_guard.py`(8), `core/prompt/gatherer_knowledge.py`(6). Mirrors §22 — the same
`CLAUDE_CHANGELOG.md` batch-narrative style inlined into source, with nothing pruning it as entries age.
Low-risk; doctrine note only.

## 25. `**kwargs` accepted, never referenced
app=12, tests=246, scripts=2. App instances low/legitimate (`orchestrator.py:build_prompt`, compat-shim
`core/prompt.py:build_prompt`). Test instances are mock/fixture call-compatibility, normal practice.

## 26. Long parameter lists (>8)
app=41, tests=4. `core/agentic/tools.py:__init__`(15), `core/agentic/controller.py:__init__`(15),
`agent_branch/run.py:run_objective`(16), `core/insight/coordinator.py:__init__`(14). Same god-classes as §16.
Closure: parameter objects/dataclasses for the worst offenders. Effort M/L.

## 27. Other observed practices (≥20 instances)
Both already folded into the ranking: `os.getenv` bypassing config (§9, promoted to #2) and the
`hasattr`-lazy-init idiom (§13, promoted to #8).

## Ranked top-10 (count × risk)
1. **Silent `except Exception` swallows** (§2) — 412 app instances, ~10x the documented degradation paths;
   matches BC-69/BC-70 (silent ops failures — Backblaze dead 46 days, daily-notes timer dead 7.5 months).
   Chokepoint/doctrine. Effort M.
2. **`os.getenv`/`os.environ` scattered outside `config/`** (§9) — 252 calls, 186 names bypassing the
   project's own config chokepoint; sibling of BC-12. Effort L.
3. **God-classes / oversized functions** (§16, §26) — 13 files >2000 lines, 13 classes >1000 lines, 80
   functions >150 lines, 41 long-param-list functions, same ~10 files behind most of the catalog's
   per-turn incidents. Effort L.
4. **Naive `datetime.now()`** (§10) — 550 naive sites vs 30 tz-aware despite two purpose-built tz modules
   built from live incidents; candidate new class below. Effort L.
5. **Try-block-as-control-flow density** (§3) — 1831 app try blocks, 67 functions >50% try-body, same file
   cluster as #1. Effort M.
6. **Eager f-string logger calls** (§23) — 2031 vs 555 lazy in app; mechanical, cheap. Effort S.
7. **Duplicate small helpers** (§15) — 34 identical-AST groups incl. 2 same-file duplicate defs and
   `_daemon_running` reimplemented in 17 scripts (BC-58 sibling). Effort S per helper.
8. **`if not hasattr(self, "_x")` lazy-init** (§13) — 103 app instances, no shared helper. Effort M.
9. **Widespread `print(`** outside documented CLI paths (§5) — 260 app instances. Effort S.
10. **Dated test files / dated source comments** (§22, §24) — 53 files / 1168 tests + 139 dated comment
    lines; intentional but fragments subject-based regression coverage. Effort L if consolidated.

(Zero/well-controlled, reported not ranked: mutable defaults §7 — 0; direct JSON-store writes §20 — 0
detected; enum-to-string §21 — 0 live after triage, class closed; TODO/FIXME §18 — 6 real; `sys.path` in
app §8 — 1, documented.)

## CANDIDATE BUG CLASSES (not in `docs/BUG_CLASSES.md`)
1. **Config-bypass via direct env read** — a setting read with `os.getenv`/`os.environ.get` directly (186
   names, 252 sites) has no YAML default, no `config.local.yaml` override, no schema validation, and can't
   be moved by a live Settings change. Distinct from BC-11 (setter fails to reach a built consumer) and
   BC-12 (a config key never reaches its reader): here no config key exists at all. Sample:
   `memory/shutdown_processor.py:22`, `knowledge/WikiManager.py:16`, `utils/query_checker.py:13`.
2. **Naive-datetime arithmetic despite an existing tz-aware helper** — 550 direct `datetime.now()`/`.utcnow()`
   sites alongside two purpose-built tz modules that exist because naive-datetime bugs already shipped
   (Sep-4 deadline, Sep-5 pacing). Distinct from BC-58: the helper EXISTS, it's simply not the default way
   to get the time. Sample: `core/agentic/controller.py` (3), `core/context_pipeline.py` (4).

## MEASUREMENT LIMITS
- Files with a syntax error are silently skipped by the AST pass, not flagged (none observed).
- "Swallow-only" except detection is a single-statement-body heuristic; a two-statement discard (e.g.
  `logger.debug(...); return None`) is undercounted as "other handling".
- Duplicate-function detection (§15) needs an EXACT `ast.dump` body match; a single reordered/renamed
  statement defeats it — counts are a lower bound.
- Enum-to-string §21's "0 live" conclusion rests on a 25-line sample + targeted re-grep, not an exhaustive
  read of all 1046 raw hits. Env-var-name extraction (§9) only sees literal string constants. `.encode(`
  async-blocking detection is name-based only, not verified against SentenceTransformer.
- `eval/` counted as app per the task's package list, though several `eval/run_phase*.py` behave like CLI
  scripts (relevant to §5). Top-N file lists are capped at the exported top 15-20 (totals are exact; JSON
  holds more of the tail). No import/execution of application modules occurred.

Outputs: `scratchpad/antipattern_survey.md`, `scratchpad/antipattern_counts.json` (both under
`/tmp/claude-1000/-home-lukeh-Daemon-v1/9b66ebf9-0f23-404f-8567-01d4d95793f8/scratchpad/`).
