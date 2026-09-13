# Baseline review and schema-2 migration — 2026-09-13

Phase 3 of `PLAN_20260913_class_guard_completion.md`. Every legacy baseline
occurrence and every candidate newly exposed by contract v2 was reviewed
against its current source and its detector's declared semantics. The
machine-readable companion, [`baseline_review_20260913.json`](baseline_review_20260913.json),
holds each occurrence's legacy key and ordinal, v2 anchor and ordinal, line
span, full source expression, source SHA-256, assessment and rationale.

## Method and limits

- Base tree `328a8ec`; reviewer: Claude Opus 5, class-guard execution session (owner review pending before commit).
- Bounded structural review. Each candidate was read in its enclosing try
  block or statement, with the function signature and docstring. Consumers
  were not traced exhaustively, so `product_risk` means a plausible class
  instance worth fixing, not a reproduced defect. No live store, provider, or
  paid model was used.
- Every occurrence stays `accepted_debt`. None is `confirmed_fixed`: no
  Plan 2 fix exists yet. None is `false_positive`: that status requires a
  detector change that makes the finding disappear from the real scan, and
  uncertainty was never resolved that way. The `assessment` field records the
  review judgment separately from the ledger status.
- Every `product_risk` occurrence names an immutable request packet under
  [`requests/`](requests/). It stays accepted debt until the source owner's
  response is integrated and a real scan shows the anchor gone.

## Migration proof (reviewed signoff)

| Invariant | Result |
|---|---|
| Legacy schema-1 baseline SHA-256 | `9bb56e6d3f072cce5becd97293aaac93f97f7a0f592b212f9579a5f6dc47a756` |
| Legacy occurrences / unique keys / duplicate extras | 133 / 129 / 4 |
| Legacy occurrences mapped one-to-one to a v2 anchor | 133 (multiset equal: True; pending: 0) |
| New candidates exposed by contract v2 (all DM-01) | 19 |
| Proposed schema-2 occurrences / unique anchors | 152 / 148 |
| accepted_debt records == baseline occurrences | 152 == 152 (per-anchor multiset equal) |
| Legacy history in the ledger re-renders to the legacy bytes | yes (SHA-256 above) |

The four legacy duplicate keys (all DM-18) map to distinct v2 anchors,
because their handlers differ, and each keeps its own legacy ordinal. The
four DM-17 script rows had legacy symbol `""`; their v2 symbol is `<module>`.
One DM-17 anchor keeps multiplicity 2: the two `"data/"` literals in
`tests/test_git_memory.py`, which were separate legacy lines.

## Summary

| Scanner | product_risk | reviewed_benign | uncertain | Total |
|---|---:|---:|---:|---:|
| DM-01 | 17 | 14 | 0 | 31 |
| DM-17 | 4 | 35 | 0 | 39 |
| DM-18 | 59 | 11 | 9 | 79 |
| DM-31 | 0 | 3 | 0 | 3 |
| **All** | 80 | 63 | 9 | 152 |

## Requests

| Request | Title | Occurrences |
|---|---|---:|
| [CGR-20260913-001](requests/CGR-20260913-001.md) | Unguarded --apply store scripts | 4 |
| [CGR-20260913-002](requests/CGR-20260913-002.md) | Agentic gate keyword tests outside utils.trigger_match | 7 |
| [CGR-20260913-003](requests/CGR-20260913-003.md) | Tone-detector keyword tests outside utils.trigger_match | 2 |
| [CGR-20260913-004](requests/CGR-20260913-004.md) | Safety canary compares the string form of a tone enum | 1 |
| [CGR-20260913-005](requests/CGR-20260913-005.md) | Query-routing keyword lists outside utils.trigger_match | 4 |
| [CGR-20260913-006](requests/CGR-20260913-006.md) | Git commit tags assigned by raw substring | 3 |
| [CGR-20260913-007](requests/CGR-20260913-007.md) | Prompt gatherers collapse retrieval failure into an empty section | 22 |
| [CGR-20260913-008](requests/CGR-20260913-008.md) | Knowledge retrieval managers return empty on store or provider errors | 11 |
| [CGR-20260913-009](requests/CGR-20260913-009.md) | Memory store reads return empty on failure | 20 |
| [CGR-20260913-010](requests/CGR-20260913-010.md) | Memory store writes return None on failure | 6 |

## Occurrences

| # | Scanner | Origin | Path | Symbol | Kind | Anchor digest | Ord | Line | Assessment | Request |
|---:|---|---|---|---|---|---|---:|---:|---|---|
| 1 | DM-01 | new (v2) | `core/actions/google_auth.py` | `GoogleAuthManager.authenticate._run_flow` | raw_membership | `59f2d5d6639d` | 1 | 170 | reviewed_benign | — |
| 2 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `2c1438c716fe` | 1 | 713 | reviewed_benign | — |
| 3 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `c73ff0856369` | 1 | 724 | product_risk | CGR-20260913-002 |
| 4 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `c73ff0856369` | 2 | 770 | product_risk | CGR-20260913-002 |
| 5 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `bc595068cda1` | 1 | 973 | product_risk | CGR-20260913-002 |
| 6 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `4a720fec20a1` | 1 | 1080 | product_risk | CGR-20260913-002 |
| 7 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `c73ff0856369` | 3 | 1107 | product_risk | CGR-20260913-002 |
| 8 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `c73ff0856369` | 4 | 1185 | product_risk | CGR-20260913-002 |
| 9 | DM-01 | new (v2) | `core/agentic/gate.py` | `evaluate_agentic_gate` | raw_membership | `bd46f63251b2` | 1 | 1514 | product_risk | CGR-20260913-002 |
| 10 | DM-01 | new (v2) | `core/insight/coordinator.py` | `_event_from_row` | raw_membership | `579321e2d3e7` | 1 | 202 | reviewed_benign | — |
| 11 | DM-01 | legacy #1 | `core/orchestrator.py` | `DaemonOrchestrator._build_system_prompt` | raw_membership | `e50f717251cf` | 1 | 1286 | reviewed_benign | — |
| 12 | DM-01 | legacy #1 | `core/prompt/formatter.py` | `_format_session_header` | raw_membership | `14401ee69b65` | 1 | 287 | reviewed_benign | — |
| 13 | DM-01 | legacy #1 | `core/prompt/formatter.py` | `_format_session_header` | raw_membership | `e9cbed17156a` | 1 | 289 | reviewed_benign | — |
| 14 | DM-01 | new (v2) | `core/safety_canary.py` | `SafetyCanary._is_conversational` | raw_membership | `61868d586a02` | 1 | 49 | product_risk | CGR-20260913-004 |
| 15 | DM-01 | legacy #1 | `knowledge/git_memory.py` | `GitMemoryExtractor._extract_tags` | raw_membership | `535a7846dd33` | 1 | 364 | product_risk | CGR-20260913-006 |
| 16 | DM-01 | legacy #1 | `knowledge/git_memory.py` | `GitMemoryExtractor._extract_tags` | raw_membership | `5a705170c0d3` | 1 | 366 | product_risk | CGR-20260913-006 |
| 17 | DM-01 | legacy #1 | `knowledge/git_memory.py` | `GitMemoryExtractor._extract_tags` | raw_membership | `b2b6e9cc4bfa` | 1 | 368 | product_risk | CGR-20260913-006 |
| 18 | DM-01 | legacy #1 | `knowledge/sandbox_manager.py` | `_kill_sandbox_quietly` | raw_membership | `6e9ef52dd8b1` | 1 | 78 | reviewed_benign | — |
| 19 | DM-01 | legacy #1 | `knowledge/web_search_manager.py` | `WebSearchManager.decompose_query` | raw_membership | `9d5df111d4da` | 1 | 2029 | reviewed_benign | — |
| 20 | DM-01 | new (v2) | `memory/corpus_manager.py` | `CorpusManager.prune_corpus` | raw_membership | `f0f0941d017a` | 1 | 464 | reviewed_benign | — |
| 21 | DM-01 | legacy #1 | `memory/memory_retriever.py` | `MemoryRetriever._maybe_temporal_window_rerank` | raw_membership | `1ed687b5c34c` | 1 | 1693 | product_risk | CGR-20260913-005 |
| 22 | DM-01 | legacy #1 | `memory/memory_scorer.py` | `MemoryScorer.rank_memories` | raw_membership | `806b5a858f14` | 1 | 514 | reviewed_benign | — |
| 23 | DM-01 | legacy #1 | `memory/user_profile.py` | `UserProfile._is_temporal_query` | raw_membership | `031e1c0068a0` | 1 | 686 | product_risk | CGR-20260913-005 |
| 24 | DM-01 | new (v2) | `scripts/latency_rollup.py` | `main` | raw_membership | `4797e2348b5e` | 1 | 80 | reviewed_benign | — |
| 25 | DM-01 | legacy #1 | `scripts/migrate_proposals_supervision.py` | `infer_test_files` | raw_membership | `1babbc662dbd` | 1 | 106 | reviewed_benign | — |
| 26 | DM-01 | new (v2) | `scripts/test_reflection_self_rewrite.py` | `main` | raw_membership | `e4c6560c5cab` | 1 | 162 | reviewed_benign | — |
| 27 | DM-01 | new (v2) | `scripts/test_reflection_self_rewrite.py` | `main` | raw_membership | `fc9f694ac1a2` | 1 | 163 | reviewed_benign | — |
| 28 | DM-01 | new (v2) | `utils/query_checker.py` | `has_thread_break_marker` | raw_membership | `6eec6b756d47` | 1 | 1436 | product_risk | CGR-20260913-005 |
| 29 | DM-01 | new (v2) | `utils/tone_detector.py` | `_check_observational_language` | raw_membership | `f35eed657914` | 1 | 941 | product_risk | CGR-20260913-003 |
| 30 | DM-01 | new (v2) | `utils/tone_detector.py` | `_calculate_harm_score` | raw_membership | `fcc1e1114001` | 1 | 1023 | product_risk | CGR-20260913-003 |
| 31 | DM-01 | new (v2) | `utils/web_search_trigger.py` | `quick_prefilter_should_skip` | raw_membership | `f50b50809eec` | 1 | 1178 | product_risk | CGR-20260913-005 |
| 32 | DM-17 | legacy #1 | `scripts/cleanup_stale_illness.py` | `<module>` | apply_without_guard | `c56e533e7ca5` | 1 | 155 | product_risk | CGR-20260913-001 |
| 33 | DM-17 | legacy #1 | `scripts/graph_relation_normalize.py` | `<module>` | apply_without_guard | `900dc03ef093` | 1 | 121 | product_risk | CGR-20260913-001 |
| 34 | DM-17 | legacy #1 | `scripts/reclassify_proposals.py` | `<module>` | apply_without_guard | `4bc72ea01f23` | 1 | 114 | product_risk | CGR-20260913-001 |
| 35 | DM-17 | legacy #1 | `scripts/restore_backup.py` | `<module>` | apply_without_guard | `537b72a4baca` | 1 | 135 | product_risk | CGR-20260913-001 |
| 36 | DM-17 | legacy #1 | `tests/agent_branch/test_coding_worker.py` | `test_repo_map_lists_files_and_skips_heavy_dirs` | test_data_path_literal | `a16a329f496e` | 1 | 51 | reviewed_benign | — |
| 37 | DM-17 | legacy #1 | `tests/agent_branch/test_manifest.py` | `test_default_forbidden_paths_cover_safety_surface` | test_data_path_literal | `4e6f5396343f` | 1 | 75 | reviewed_benign | — |
| 38 | DM-17 | legacy #1 | `tests/test_eval/test_persistence_guard.py` | `TestPersistenceSnapshot._make_snapshot` | test_data_path_literal | `d7210dc03b51` | 1 | 40 | reviewed_benign | — |
| 39 | DM-17 | legacy #1 | `tests/test_eval/test_persistence_guard.py` | `TestPersistenceSnapshot.test_changed_file_hash_fails` | test_data_path_literal | `d7210dc03b51` | 1 | 72 | reviewed_benign | — |
| 40 | DM-17 | legacy #1 | `tests/test_eval/test_persistence_guard.py` | `TestPersistenceSnapshot.test_changed_file_size_fails` | test_data_path_literal | `d7210dc03b51` | 1 | 83 | reviewed_benign | — |
| 41 | DM-17 | legacy #1 | `tests/test_eval/test_persistence_guard.py` | `TestPersistenceSnapshot.test_diff_reports_all_changes` | test_data_path_literal | `d7210dc03b51` | 1 | 120 | reviewed_benign | — |
| 42 | DM-17 | legacy #1 | `tests/test_git_memory.py` | `TestHotFiles.test_exclude_globs_filters` | test_data_path_literal | `4e6f5396343f` | 1 | 225 | reviewed_benign | — |
| 43 | DM-17 | legacy #1 | `tests/test_git_memory.py` | `TestHotFiles.test_exclude_globs_filters` | test_data_path_literal | `4e6f5396343f` | 2 | 227 | reviewed_benign | — |
| 44 | DM-17 | legacy #1 | `tests/unit/test_fs_snapshot.py` | `TestShouldExclude.test_backup_dir_excluded` | test_data_path_literal | `f7fb143914c6` | 1 | 65 | reviewed_benign | — |
| 45 | DM-17 | legacy #1 | `tests/unit/test_fs_snapshot.py` | `TestShouldExclude.test_backup_dir_excluded` | test_data_path_literal | `a202ac4b61cf` | 1 | 66 | reviewed_benign | — |
| 46 | DM-17 | legacy #1 | `tests/unit/test_fs_snapshot.py` | `TestShouldExclude.test_backup_dir_excluded` | test_data_path_literal | `c986574ca135` | 1 | 67 | reviewed_benign | — |
| 47 | DM-17 | legacy #1 | `tests/unit/test_fs_snapshot.py` | `TestShouldExclude.test_other_data_files_still_included` | test_data_path_literal | `2cb9f3f2b2e0` | 1 | 70 | reviewed_benign | — |
| 48 | DM-17 | legacy #1 | `tests/unit/test_python_fs_guard.py` | `TestPathResolution.test_nested_path` | test_data_path_literal | `d2abd14642c5` | 1 | 502 | reviewed_benign | — |
| 49 | DM-17 | legacy #1 | `tests/unit/test_python_fs_guard.py` | `TestProtectedPathChecks.test_protected_subpath` | test_data_path_literal | `d2abd14642c5` | 1 | 689 | reviewed_benign | — |
| 50 | DM-17 | legacy #1 | `tests/unit/test_python_fs_guard.py` | `TestProtectedPathChecks.test_deeply_nested` | test_data_path_literal | `3a7c76396b55` | 1 | 701 | reviewed_benign | — |
| 51 | DM-17 | legacy #1 | `tests/unit/test_python_fs_guard.py` | `TestProtectedPathChecks.test_normal_path_not_always_blocked` | test_data_path_literal | `616307e7af98` | 1 | 713 | reviewed_benign | — |
| 52 | DM-17 | legacy #1 | `tests/unit/test_retrospective_small_guards.py` | `TestDaemonStateFsExemption.test_exemption_covers_tmp_sibling` | test_data_path_literal | `be8a1ebcd79b` | 1 | 135 | reviewed_benign | — |
| 53 | DM-17 | legacy #1 | `tests/unit/test_retrospective_small_guards.py` | `TestDaemonStateFsExemption.test_other_data_paths_not_exempt` | test_data_path_literal | `114deb0f6943` | 1 | 140 | reviewed_benign | — |
| 54 | DM-17 | legacy #1 | `tests/unit/test_retrospective_small_guards.py` | `TestDaemonStateFsExemption.test_other_data_paths_not_exempt` | test_data_path_literal | `88b9e5dc55ff` | 1 | 141 | reviewed_benign | — |
| 55 | DM-17 | legacy #1 | `tests/unit/test_retrospective_small_guards.py` | `TestDaemonStateFsExemption.test_check_and_maybe_block_allows_credits_in_agent_mode` | test_data_path_literal | `be8a1ebcd79b` | 1 | 156 | reviewed_benign | — |
| 56 | DM-17 | legacy #1 | `tests/unit/test_retrospective_small_guards.py` | `TestDaemonStateFsExemption.test_check_and_maybe_block_allows_credits_in_agent_mode` | test_data_path_literal | `2cb9f3f2b2e0` | 1 | 160 | reviewed_benign | — |
| 57 | DM-17 | legacy #1 | `tests/unit/test_sep10_calendar_forced_action_loop.py` | `TestFsGuardTempSibling.test_pending_store_temp_sibling_is_exempt` | test_data_path_literal | `7dfc866cf630` | 1 | 235 | reviewed_benign | — |
| 58 | DM-17 | legacy #1 | `tests/unit/test_sep10_calendar_forced_action_loop.py` | `TestFsGuardTempSibling.test_pending_store_temp_sibling_is_exempt` | test_data_path_literal | `6e60c158ff20` | 1 | 236 | reviewed_benign | — |
| 59 | DM-17 | legacy #1 | `tests/unit/test_sep10_calendar_forced_action_loop.py` | `TestFsGuardTempSibling.test_pending_store_temp_sibling_is_exempt` | test_data_path_literal | `8abf38b2c4a6` | 1 | 237 | reviewed_benign | — |
| 60 | DM-17 | legacy #1 | `tests/unit/test_sep10_calendar_forced_action_loop.py` | `TestFsGuardTempSibling.test_pending_store_temp_sibling_is_exempt` | test_data_path_literal | `4bf3df7aab0b` | 1 | 238 | reviewed_benign | — |
| 61 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestRmProtected.test_rm_rf_data_subdir` | test_data_path_literal | `d992fa3151e3` | 1 | 184 | reviewed_benign | — |
| 62 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestRmProtected.test_rm_data_file_inside_protected` | test_data_path_literal | `2cb9f3f2b2e0` | 1 | 220 | reviewed_benign | — |
| 63 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestRmProtected.test_rm_path_traversal` | test_data_path_literal | `d068c67cb53e` | 1 | 231 | reviewed_benign | — |
| 64 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestMvProtected.test_mv_data_subpath` | test_data_path_literal | `30d578c3440c` | 1 | 262 | reviewed_benign | — |
| 65 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestTruncateProtected.test_truncate_data_file` | test_data_path_literal | `2cb9f3f2b2e0` | 1 | 363 | reviewed_benign | — |
| 66 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestResolveTarget.test_nested_path` | test_data_path_literal | `e529fe8374ab` | 1 | 387 | reviewed_benign | — |
| 67 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestResolveTarget.test_traversal_stays_in_repo` | test_data_path_literal | `d068c67cb53e` | 1 | 400 | reviewed_benign | — |
| 68 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestProtectedPaths.test_subpath_of_protected_dir` | test_data_path_literal | `d992fa3151e3` | 1 | 427 | reviewed_benign | — |
| 69 | DM-17 | legacy #1 | `tests/unit/test_shell_cmd_guard.py` | `TestProtectedPaths.test_deeply_nested_protected` | test_data_path_literal | `271c1d7df224` | 1 | 439 | reviewed_benign | — |
| 70 | DM-17 | legacy #1 | `tests/unit/test_tone_floor_self_latch.py` | `TestStateSandbox.test_tone_state_path_is_sandboxed_here` | test_data_path_literal | `4e6f5396343f` | 1 | 126 | reviewed_benign | — |
| 71 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_personal_notes` | broad_except_returns_empty | `0aceb2a32d20` | 1 | 721 | product_risk | CGR-20260913-007 |
| 72 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_reference_docs` | broad_except_returns_empty | `1bf445bf8ac5` | 1 | 789 | product_risk | CGR-20260913-007 |
| 73 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin._fetch_upload_roster` | broad_except_returns_empty | `65dc7fab931b` | 1 | 822 | product_risk | CGR-20260913-007 |
| 74 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_user_uploads` | broad_except_returns_empty | `0f7e8beca882` | 1 | 998 | product_risk | CGR-20260913-007 |
| 75 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_git_commits` | broad_except_returns_empty | `080e350f58e7` | 1 | 1093 | product_risk | CGR-20260913-007 |
| 76 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_proposed_features` | broad_except_returns_empty | `f569c8b02c03` | 1 | 1148 | product_risk | CGR-20260913-007 |
| 77 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_procedural_skills` | broad_except_returns_empty | `a2a5682bc707` | 1 | 1191 | product_risk | CGR-20260913-007 |
| 78 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_graph_context` | broad_except_returns_empty | `90846424a576` | 1 | 1253 | product_risk | CGR-20260913-007 |
| 79 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_unresolved_threads` | broad_except_returns_empty | `54b28ae2d6ae` | 1 | 1280 | product_risk | CGR-20260913-007 |
| 80 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin._get_wiki_content_timed` | broad_except_returns_empty | `1f054e9dd502` | 1 | 1826 | product_risk | CGR-20260913-007 |
| 81 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_narrative_context` | broad_except_returns_empty | `b970de35eff0` | 1 | 2000 | product_risk | CGR-20260913-007 |
| 82 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_daemon_self_notes` | broad_except_returns_empty | `9c485c8e097e` | 1 | 2065 | product_risk | CGR-20260913-007 |
| 83 | DM-18 | legacy #1 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_relevant_emails` | broad_except_returns_empty | `285fb9455536` | 1 | 2257 | product_risk | CGR-20260913-007 |
| 84 | DM-18 | legacy #2 | `core/prompt/gatherer_knowledge.py` | `KnowledgeRetrievalMixin.get_relevant_emails` | broad_except_returns_empty | `fb824ccfe79c` | 1 | 2261 | product_risk | CGR-20260913-007 |
| 85 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin.get_recent_facts` | broad_except_returns_empty | `11ecd20abdf9` | 1 | 155 | product_risk | CGR-20260913-007 |
| 86 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin.get_facts` | broad_except_returns_empty | `5ec5e4cecd6b` | 1 | 167 | product_risk | CGR-20260913-007 |
| 87 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin._get_recent_conversations` | broad_except_returns_empty | `ee96452023c1` | 1 | 229 | product_risk | CGR-20260913-007 |
| 88 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin._get_semantic_memories` | broad_except_returns_empty | `b0785fd88c03` | 1 | 634 | product_risk | CGR-20260913-007 |
| 89 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin._get_reflections` | broad_except_returns_empty | `6c27f1f6f296` | 1 | 799 | product_risk | CGR-20260913-007 |
| 90 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin.get_user_profile_context` | broad_except_returns_empty | `7aad109e494d` | 1 | 964 | product_risk | CGR-20260913-007 |
| 91 | DM-18 | legacy #1 | `core/prompt/gatherer_memory.py` | `MemoryRetrievalMixin.get_upcoming_schedule` | broad_except_returns_empty | `1b8a2bd0a425` | 1 | 1070 | product_risk | CGR-20260913-007 |
| 92 | DM-18 | legacy #1 | `core/prompt/gatherer_web.py` | `WebSearchMixin._get_web_search_results` | broad_except_returns_empty | `495fad3f30ea` | 1 | 304 | product_risk | CGR-20260913-007 |
| 93 | DM-18 | legacy #1 | `knowledge/WikiManager.py` | `WikiManager._fetch_extract_action_api` | broad_except_returns_empty | `cd5800228d2c` | 1 | 344 | reviewed_benign | — |
| 94 | DM-18 | legacy #1 | `knowledge/WikiManager.py` | `WikiManager.resolve_and_fetch` | broad_except_returns_empty | `d26e54b6ea04` | 1 | 633 | uncertain | — |
| 95 | DM-18 | legacy #1 | `knowledge/clip_manager.py` | `CLIPManager.encode_image` | broad_except_returns_empty | `977a04df986f` | 1 | 143 | reviewed_benign | — |
| 96 | DM-18 | legacy #1 | `knowledge/clip_manager.py` | `CLIPManager.encode_text` | broad_except_returns_empty | `e5cc72d6ef15` | 1 | 173 | reviewed_benign | — |
| 97 | DM-18 | legacy #1 | `knowledge/document_generator.py` | `DocumentGenerator._search_web` | broad_except_returns_empty | `28b3d64b0d02` | 1 | 581 | uncertain | — |
| 98 | DM-18 | legacy #1 | `knowledge/document_generator.py` | `DocumentGenerator._search_collection` | broad_except_returns_empty | `be9ebe69d9d3` | 1 | 616 | uncertain | — |
| 99 | DM-18 | legacy #1 | `knowledge/graph_walk_generator.py` | `GraphWalkGenerator._articulate_and_package` | broad_except_returns_empty | `09b2c8dcf911` | 1 | 398 | reviewed_benign | — |
| 100 | DM-18 | legacy #1 | `knowledge/implementation_detector.py` | `ImplementationDetector._stage_llm_judgment` | broad_except_returns_empty | `779fe5c8c6c6` | 1 | 410 | reviewed_benign | — |
| 101 | DM-18 | legacy #1 | `knowledge/obsidian_manager.py` | `ObsidianManager.get_notes` | broad_except_returns_empty | `aae847842dee` | 1 | 781 | product_risk | CGR-20260913-008 |
| 102 | DM-18 | legacy #1 | `knowledge/obsidian_manager.py` | `ObsidianManager._keyword_search` | broad_except_returns_empty | `361ac0cb6539` | 1 | 927 | product_risk | CGR-20260913-008 |
| 103 | DM-18 | legacy #1 | `knowledge/reference_docs_manager.py` | `ReferenceDocsManager._get_document_chunks` | broad_except_returns_empty | `22cdf7957462` | 1 | 509 | product_risk | CGR-20260913-008 |
| 104 | DM-18 | legacy #1 | `knowledge/reference_docs_manager.py` | `ReferenceDocsManager.get_documents` | broad_except_returns_empty | `281aa17cae55` | 1 | 718 | product_risk | CGR-20260913-008 |
| 105 | DM-18 | legacy #1 | `knowledge/reference_docs_manager.py` | `ReferenceDocsManager._keyword_search` | broad_except_returns_empty | `bb870d883847` | 1 | 818 | product_risk | CGR-20260913-008 |
| 106 | DM-18 | legacy #1 | `knowledge/reference_docs_manager.py` | `ReferenceDocsManager.list_documents` | broad_except_returns_empty | `160602899170` | 1 | 855 | product_risk | CGR-20260913-008 |
| 107 | DM-18 | legacy #1 | `knowledge/semantic_search.py` | `SemanticSearchIndex.search` | broad_except_returns_empty | `acecd33f773a` | 1 | 347 | product_risk | CGR-20260913-008 |
| 108 | DM-18 | legacy #2 | `knowledge/semantic_search.py` | `SemanticSearchIndex.search` | broad_except_returns_empty | `24af3eabcf4d` | 1 | 378 | product_risk | CGR-20260913-008 |
| 109 | DM-18 | legacy #1 | `knowledge/visual_memory_store.py` | `VisualMemoryStore.search_by_text` | broad_except_returns_empty | `d56840dd4b31` | 1 | 281 | product_risk | CGR-20260913-008 |
| 110 | DM-18 | legacy #1 | `knowledge/web_search_manager.py` | `WebSearchManager._tavily_search` | broad_except_returns_empty | `b3d096f91950` | 1 | 1538 | product_risk | CGR-20260913-008 |
| 111 | DM-18 | legacy #1 | `knowledge/web_search_manager.py` | `WebSearchManager._tavily_extract` | broad_except_returns_empty | `e7408f555aeb` | 1 | 1744 | product_risk | CGR-20260913-008 |
| 112 | DM-18 | legacy #1 | `knowledge/web_search_manager.py` | `WebSearchManager._select_links_for_following` | broad_except_returns_empty | `af4f65e1faf3` | 1 | 1810 | reviewed_benign | — |
| 113 | DM-18 | legacy #1 | `memory/context_surfacer.py` | `ContextSurfacer.generate_insights` | broad_except_returns_empty | `e0a0745865e5` | 1 | 194 | reviewed_benign | — |
| 114 | DM-18 | legacy #1 | `memory/curation/engine.py` | `CurationEngine._store_size` | broad_except_returns_empty | `06b14c5d3804` | 1 | 348 | reviewed_benign | — |
| 115 | DM-18 | legacy #1 | `memory/llm_fact_extractor.py` | `LLMFactExtractor.extract_triples` | broad_except_returns_empty | `22f9437f13b7` | 1 | 578 | uncertain | — |
| 116 | DM-18 | legacy #2 | `memory/llm_fact_extractor.py` | `LLMFactExtractor.extract_triples` | broad_except_returns_empty | `d9296a83efb7` | 1 | 609 | uncertain | — |
| 117 | DM-18 | legacy #1 | `memory/memory_consolidator.py` | `MemoryConsolidator._current_status_facts` | broad_except_returns_empty | `1151b107e66b` | 1 | 216 | product_risk | CGR-20260913-009 |
| 118 | DM-18 | legacy #1 | `memory/memory_consolidator.py` | `MemoryConsolidator._read_obsidian_weekly_summaries` | broad_except_returns_empty | `a699b4ede0dc` | 1 | 490 | product_risk | CGR-20260913-009 |
| 119 | DM-18 | legacy #1 | `memory/memory_consolidator.py` | `MemoryConsolidator._read_obsidian_monthly_summaries` | broad_except_returns_empty | `fdecc0e254df` | 1 | 552 | product_risk | CGR-20260913-009 |
| 120 | DM-18 | legacy #1 | `memory/memory_consolidator.py` | `MemoryConsolidator._read_obsidian_daily_notes` | broad_except_returns_empty | `f7d0ec66bac7` | 1 | 618 | product_risk | CGR-20260913-009 |
| 121 | DM-18 | legacy #1 | `memory/memory_coordinator.py` | `MemoryCoordinator.get_unresolved_threads` | broad_except_returns_empty | `0f191e1ee4c9` | 1 | 600 | product_risk | CGR-20260913-009 |
| 122 | DM-18 | legacy #1 | `memory/memory_expander.py` | `MemoryExpander._fetch_conversations_in_range` | broad_except_returns_empty | `9b408d15cb41` | 1 | 365 | product_risk | CGR-20260913-009 |
| 123 | DM-18 | legacy #1 | `memory/memory_retriever.py` | `_metadata_fallback_search` | broad_except_returns_empty | `4f0488a09ecb` | 1 | 407 | product_risk | CGR-20260913-009 |
| 124 | DM-18 | legacy #1 | `memory/memory_retriever.py` | `MemoryRetriever._keyword_anchor_memories` | broad_except_returns_empty | `ca03b03b3a46` | 1 | 592 | reviewed_benign | — |
| 125 | DM-18 | legacy #1 | `memory/memory_retriever.py` | `MemoryRetriever.get_recent_facts` | broad_except_returns_empty | `7188f287c0d5` | 1 | 646 | product_risk | CGR-20260913-009 |
| 126 | DM-18 | legacy #1 | `memory/memory_retriever.py` | `MemoryRetriever.get_skills` | broad_except_returns_empty | `fe0317e81c78` | 1 | 1184 | product_risk | CGR-20260913-009 |
| 127 | DM-18 | legacy #1 | `memory/memory_storage.py` | `MemoryStorage.store_interaction` | broad_except_returns_empty | `e0ee8b4e4bf7` | 1 | 1024 | product_risk | CGR-20260913-010 |
| 128 | DM-18 | legacy #1 | `memory/memory_storage.py` | `MemoryStorage.store_skill` | broad_except_returns_empty | `741ef36f9c28` | 1 | 1516 | product_risk | CGR-20260913-010 |
| 129 | DM-18 | legacy #2 | `memory/memory_storage.py` | `MemoryStorage.store_skill` | broad_except_returns_empty | `56dcad07c9dd` | 1 | 1550 | product_risk | CGR-20260913-010 |
| 130 | DM-18 | legacy #1 | `memory/memory_storage.py` | `MemoryStorage._get_recent_summaries_by_timespan` | broad_except_returns_empty | `7bede00056b9` | 1 | 1650 | product_risk | CGR-20260913-009 |
| 131 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.store_proposal` | broad_except_returns_empty | `5b5f1d2297c9` | 1 | 103 | product_risk | CGR-20260913-010 |
| 132 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.query_proposals` | broad_except_returns_empty | `9ccced1ec502` | 1 | 163 | product_risk | CGR-20260913-009 |
| 133 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.get_proposal` | broad_except_returns_empty | `ca4b6597701c` | 1 | 184 | product_risk | CGR-20260913-009 |
| 134 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.get_pending` | broad_except_returns_empty | `55053de77902` | 1 | 204 | product_risk | CGR-20260913-009 |
| 135 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.check_similarity` | broad_except_returns_empty | `fb8608421178` | 1 | 348 | uncertain | — |
| 136 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.get_pending_and_approved` | broad_except_returns_empty | `3b129b9bc0ba` | 1 | 413 | product_risk | CGR-20260913-009 |
| 137 | DM-18 | legacy #1 | `memory/proposal_store.py` | `ProposalStore.get_for_dedup` | broad_except_returns_empty | `8aae9b14a8ec` | 1 | 449 | uncertain | — |
| 138 | DM-18 | legacy #1 | `memory/shutdown_processor.py` | `ShutdownProcessor._extract_procedural_skills` | broad_except_returns_empty | `64099756bd18` | 1 | 1127 | reviewed_benign | — |
| 139 | DM-18 | legacy #1 | `memory/storage/multi_collection_chroma_store.py` | `MultiCollectionChromaStore.get_by_id` | broad_except_returns_empty | `06b14c5d3804` | 1 | 510 | product_risk | CGR-20260913-009 |
| 140 | DM-18 | legacy #1 | `memory/storage/multi_collection_chroma_store.py` | `MultiCollectionChromaStore.add_conversation_memory` | broad_except_returns_empty | `c2aa1370af50` | 1 | 560 | product_risk | CGR-20260913-010 |
| 141 | DM-18 | legacy #1 | `memory/synthesis_memory.py` | `SynthesisMemory.find_similar` | broad_except_returns_empty | `cacb9917f5db` | 1 | 99 | product_risk | CGR-20260913-009 |
| 142 | DM-18 | legacy #1 | `memory/synthesis_memory.py` | `SynthesisMemory.get_recurring` | broad_except_returns_empty | `16b4483f4db5` | 1 | 208 | product_risk | CGR-20260913-009 |
| 143 | DM-18 | legacy #1 | `memory/synthesis_memory.py` | `SynthesisMemory.get_all_results` | broad_except_returns_empty | `fe6c2edd0ea4` | 1 | 435 | product_risk | CGR-20260913-009 |
| 144 | DM-18 | legacy #1 | `memory/thread_extractor.py` | `ThreadExtractor.extract_new_threads` | broad_except_returns_empty | `a4bf7da64cf4` | 1 | 230 | uncertain | — |
| 145 | DM-18 | legacy #1 | `memory/thread_extractor.py` | `ThreadExtractor.detect_resolutions` | broad_except_returns_empty | `0b9a0de7d2f5` | 1 | 329 | uncertain | — |
| 146 | DM-18 | legacy #1 | `memory/thread_store.py` | `ThreadStore.store_thread` | broad_except_returns_empty | `9ab58b725d85` | 1 | 257 | product_risk | CGR-20260913-010 |
| 147 | DM-18 | legacy #1 | `memory/thread_store.py` | `ThreadStore.list_open_threads` | broad_except_returns_empty | `5bd9d94c58e4` | 1 | 277 | product_risk | CGR-20260913-009 |
| 148 | DM-18 | legacy #1 | `memory/thread_store.py` | `ThreadStore.query_threads` | broad_except_returns_empty | `54188071ef72` | 1 | 369 | product_risk | CGR-20260913-009 |
| 149 | DM-18 | legacy #1 | `memory/user_profile_schema.py` | `_get_exemplar_embeddings` | broad_except_returns_empty | `06b14c5d3804` | 1 | 721 | reviewed_benign | — |
| 150 | DM-31 | legacy #1 | `core/competitive_scorer.py` | `apply_competitive_selection` | live_state_literal_default | `06a4d5b813ae` | 1 | 22 | reviewed_benign | — |
| 151 | DM-31 | legacy #1 | `core/safety_canary.py` | `SafetyCanary.__init__` | live_state_literal_default | `9711f00d956e` | 1 | 35 | reviewed_benign | — |
| 152 | DM-31 | legacy #1 | `memory/skill_activation.py` | `SkillActivationPolicy.__init__` | live_state_literal_default | `9711f00d956e` | 1 | 136 | reviewed_benign | — |
