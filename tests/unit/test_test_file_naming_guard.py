"""Naming guard (2026-10-16, plan v2 section 4.3): tests are named by module and behaviour.

Convention (docs/TEST_LANES.md section 6): ``tests/unit/test_<module>_<behaviour>.py``; the incident date and
bug class go in the module docstring.  This guard enforces it on the WORKING TREE only (no git blob reads,
consistent with ``test_no_git_state_in_tests.py``):

* no file under ``tests/`` is date-named (``test_<month><digit>...``) except the exact, hard-coded
  ``APPROVED_EXCEPTIONS`` set (changing it is a code change here, reviewed like a ratchet - it is NOT derived
  from the map, so adding a TSV row cannot raise it);
* ``tests/RENAMES.tsv`` (the permanent old-path -> new-path resolver) is well formed: renamed rows have the
  old path absent and the new path present; exempt rows have the old path present, ``new_path`` ``-``, a
  non-empty reason and membership in ``APPROVED_EXCEPTIONS``; no duplicate old/new paths; a new path is a
  ``tests/**/test_*.py`` file that is not itself date-named; a stale exempt row (file gone) is rejected.

``PRE_MIGRATION_ALLOWLIST`` is TEMPORARY: the exact list of the date-named files that exist before the
migration commit.  Lane M's migration commit deletes it (the applier's edit set removes the block between the
BEGIN/END markers); until then it only stops NEW dated files appearing.

class: convention drift (no docs/BUG_CLASSES.md entry fits; a draft is in the lane's bc_draft.md).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DATED_RE = re.compile(rf"^test_(?:{_MONTHS})\d", re.I)

# The ONLY approved exception: it anchors records in config/bug_class_dispositions.json and
# config/bug_class_baseline.json; re-anchoring is a separate, authorized class-guard change.
APPROVED_EXCEPTIONS: frozenset[str] = frozenset({
    "tests/unit/test_sep10_calendar_forced_action_loop.py",
})

# BEGIN PRE-MIGRATION ALLOWLIST (temporary; lane M deletes this block)
PRE_MIGRATION_ALLOWLIST: frozenset[str] = frozenset({
        "tests/unit/test_oct02_active_features_truth.py",
        "tests/unit/test_oct02_frozen_paths.py",
        "tests/unit/test_oct02_household_canonical.py",
        "tests/unit/test_oct02_living_situation_durable.py",
        "tests/unit/test_oct02_profile_relevance.py",
        "tests/unit/test_oct02_proposal_store_outcomes.py",
        "tests/unit/test_oct02_residence_attribution.py",
        "tests/unit/test_oct02_residence_conflicts_curator.py",
        "tests/unit/test_oct02_residence_junk.py",
        "tests/unit/test_oct02_spec_contract.py",
        "tests/unit/test_oct02_thread_shift_continuation.py",
        "tests/unit/test_oct08_add_profile_fact_preview.py",
        "tests/unit/test_oct08_agentic_retry_receipts.py",
        "tests/unit/test_oct08_claim_time.py",
        "tests/unit/test_oct08_gather_session_state.py",
        "tests/unit/test_oct08_profile_supersession.py",
        "tests/unit/test_oct08_semantic_scheduling.py",
        "tests/unit/test_oct08_stream_spacing.py",
        "tests/unit/test_oct08_trigger_veto_order.py",
        "tests/unit/test_oct08_turn_path.py",
        "tests/unit/test_oct08b_attachment_fact_boundary.py",
        "tests/unit/test_oct08b_data_root_authority.py",
        "tests/unit/test_oct08b_prior_attachment_docgen.py",
        "tests/unit/test_oct08b_reply_offer.py",
        "tests/unit/test_oct08b_unicode_display_names.py",
        "tests/unit/test_oct08c_navigation_matcher.py",
        "tests/unit/test_oct08c_pattern_engine_utc.py",
        "tests/unit/test_oct08c_third_party_narration.py",
        "tests/unit/test_oct08d_agentic_context.py",
        "tests/unit/test_oct08d_thread_disputed.py",
        "tests/unit/test_oct08d_thread_provenance.py",
        "tests/unit/test_oct08e_explicit_search_routing.py",
        "tests/unit/test_oct08e_pending_turn_visibility.py",
        "tests/unit/test_oct08e_stm_fidelity.py",
        "tests/unit/test_oct08e_unresolved_referent_search.py",
        "tests/unit/test_oct10_anaphoric_new_referent.py",
        "tests/unit/test_oct10_fact_quality.py",
        "tests/unit/test_oct10_flush_window.py",
        "tests/unit/test_oct10_frozen_build_fixes.py",
        "tests/unit/test_oct10_frozen_store_paths.py",
        "tests/unit/test_oct10_hygiene.py",
        "tests/unit/test_oct10_sensitive_fact_gating.py",
        "tests/unit/test_oct10_shutdown_budget_paste.py",
        "tests/unit/test_oct10_thread_dispute_per_turn.py",
        "tests/unit/test_oct12_b1_prompt_pronouns.py",
        "tests/unit/test_oct12_b1_user_mode.py",
        "tests/unit/test_oct12_b2_setup_core.py",
        "tests/unit/test_oct12_b2_setup_routes.py",
        "tests/unit/test_oct12_b3_app_lifecycle.py",
        "tests/unit/test_oct12_b4_marker_wording.py",
        "tests/unit/test_oct12_b4_mood_fact_classification.py",
        "tests/unit/test_oct12_b4_notes_speaker_authority.py",
        "tests/unit/test_oct12_b4_recent_window_pastes.py",
        "tests/unit/test_oct12_b4_reflection_double_render.py",
        "tests/unit/test_oct12_b4_shutdown_fact_window.py",
        "tests/unit/test_oct12_b4_user_stated_evidence.py",
        "tests/unit/test_oct12_b5_paste_blind_routing.py",
        "tests/unit/test_oct12_b5_weather_location.py",
        "tests/unit/test_oct12_b6_retrieval_quality.py",
        "tests/unit/test_oct12_fg_fetch_provenance.py",
        "tests/unit/test_oct12_pl_polish.py",
        "tests/unit/test_oct12_r1be_problem_report.py",
        "tests/unit/test_oct12_x1_guidance_selfref.py",
        "tests/unit/test_oct12_x2_citation_drift.py",
        "tests/unit/test_oct12_x3_location_and_note_tools.py",
        "tests/unit/test_oct13_b4r_user_stated_ranking.py",
        "tests/unit/test_oct14_small_bare_city.py",
        "tests/unit/test_oct14_small_interjection_junk.py",
        "tests/unit/test_oct14_usr_candidates.py",
        "tests/unit/test_oct14e_claim_span_markers.py",
        "tests/unit/test_oct14e_paste_gate_scope.py",
        "tests/unit/test_oct14e_paste_tone_head.py",
        "tests/unit/test_oct14e_sum_coverage.py",
        "tests/unit/test_oct14e_thr_quick_resolution.py",
        "tests/unit/test_oct14e_thr_single_judge.py",
        "tests/unit/test_oct15_bf_backfill.py",
        "tests/unit/test_oct15_quar_corpus_readers.py",
        "tests/unit/test_oct15_quar_tool.py",
        "tests/unit/test_sep01_grounding_misfires.py",
        "tests/unit/test_sep02_checkin_fixes.py",
        "tests/unit/test_sep03_cat_session_fixes.py",
        "tests/unit/test_sep03_followups_continuity.py",
        "tests/unit/test_sep03_followups_gating.py",
        "tests/unit/test_sep03_followups_graph.py",
        "tests/unit/test_sep03_live_probe_fixes.py",
        "tests/unit/test_sep04_attachment_turn.py",
        "tests/unit/test_sep04_evening_fixes.py",
        "tests/unit/test_sep05_curation_wave2.py",
        "tests/unit/test_sep05_evening_turn_audit.py",
        "tests/unit/test_sep05_heavy_history_first_person.py",
        "tests/unit/test_sep05_table_preview.py",
        "tests/unit/test_sep05_thread_override_honesty.py",
        "tests/unit/test_sep05_web_cache_same_query.py",
        "tests/unit/test_sep06_conversation_response.py",
        "tests/unit/test_sep06_evidence_ownership.py",
        "tests/unit/test_sep06_retest_closures.py",
        "tests/unit/test_sep07_calendar_offer_continuation.py",
        "tests/unit/test_sep07_followups.py",
        "tests/unit/test_sep07_reuse_stm_intent.py",
        "tests/unit/test_sep08_agentic_answer_integrity.py",
        "tests/unit/test_sep08_answer_call_export.py",
        "tests/unit/test_sep08_document_continuity.py",
        "tests/unit/test_sep08_homework_tone_misfires.py",
        "tests/unit/test_sep08_lazy_sandbox.py",
        "tests/unit/test_sep08_ops_fixes.py",
        "tests/unit/test_sep08_profile_fact_misfires.py",
        "tests/unit/test_sep08_request_scope.py",
        "tests/unit/test_sep09_atomic_writers.py",
        "tests/unit/test_sep09_backup_recovery.py",
        "tests/unit/test_sep09_curation_integrity.py",
        "tests/unit/test_sep09_forced_action_type.py",
        "tests/unit/test_sep09_latency_metrics.py",
        "tests/unit/test_sep09_live_controls.py",
        "tests/unit/test_sep09_speed_batch.py",
        "tests/unit/test_sep09_speed_images.py",
        "tests/unit/test_sep09_storage_repairs.py",
        "tests/unit/test_sep10_probe_dump_actions.py",
        "tests/unit/test_sep10_probe_dump_interpretation.py",
        "tests/unit/test_sep10_referee_followups.py",
        "tests/unit/test_sep10_web_search_gap.py",
        "tests/unit/test_sep12_action_claim_eligibility.py",
        "tests/unit/test_sep12_followup_budget_outcomes.py",
        "tests/unit/test_sep12_followup_calendar_attribution.py",
        "tests/unit/test_sep12_followup_identity_names.py",
        "tests/unit/test_sep12_keyword_boundary_and_budget.py",
        "tests/unit/test_sep12_repository_status_context.py",
        "tests/unit/test_sep12_search_budget_reservation.py",
        "tests/unit/test_sep12_search_identity_scope.py",
        "tests/unit/test_sep12_web_evidence_budget.py",
        "tests/unit/test_sep20_doc_attachment_source.py",
        "tests/unit/test_sep21_response_tail.py",
        "tests/unit/test_sep22_calendar_backstop_user_words.py",
        "tests/unit/test_sep22_delivery_revision_sequence.py",
        "tests/unit/test_sep22_formatter_machinery_strip.py",
        "tests/unit/test_sep22_graph_function_words.py",
        "tests/unit/test_sep22_topup_dedup_parity.py",
        "tests/unit/test_sep27_config_paths.py",
        "tests/unit/test_sep27_email_auth_failure.py",
        "tests/unit/test_sep27_llm_fact_observed_at.py",
        "tests/unit/test_sep27_session_truth.py",
        "tests/unit/test_sep27_tool_continuation.py",
        "tests/unit/test_sep27b_B1.py",
        "tests/unit/test_sep27b_B2.py",
        "tests/unit/test_sep27b_B3.py",
        "tests/unit/test_sep27b_B4.py",
        "tests/unit/test_sep27b_B5.py",
        "tests/unit/test_sep27b_B6.py",
        "tests/unit/test_sep27b_B7.py",
        "tests/unit/test_sep27b_B8.py",
        "tests/unit/test_sep27b_B9.py",
        "tests/unit/test_sep27f_forced_read_tool.py",
        "tests/unit/test_sep27f_marker_guard_envelope.py",
        "tests/unit/test_sep28_contact_lookup_unavailable.py",
        "tests/unit/test_sep28_gate_url_detect.py",
        "tests/unit/test_sep28_google_readers_auth_failure.py",
        "tests/unit/test_sep28_offer_write_verb.py",
        "tests/unit/test_sep28_settings_save_local.py",
        "tests/unit/test_sep28_turn_record_tool_receipts.py",
        "tests/unit/test_sep30_file_access_secrets.py",
        "tests/unit/test_sep30_flush_window.py",
        "tests/unit/test_sep30_models_put_local_yaml.py",
        "tests/unit/test_sep30_notes_root.py",
        "tests/unit/test_sep30_tool_offer_truth.py",
        "tests/unit/test_sep30_tool_receipt_wrapped_empty.py",
        "tests/unit/test_sep30_user_mode_no_auto_delete.py",
})
# END PRE-MIGRATION ALLOWLIST

RENAMES_REL = "tests/RENAMES.tsv"
COLUMNS = (
    "kind", "old_path", "new_path", "subject_module", "other_subjects",
    "confidence", "evidence", "flags", "reason",
)
COLUMN_LINE = "\t".join(COLUMNS)


def _tests_files(root: Path):
    tests = root / "tests"
    for dirpath, dirnames, filenames in os.walk(tests):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            yield (Path(dirpath) / name).relative_to(root).as_posix()


def check_naming(root: Path, *, exceptions: frozenset[str] = APPROVED_EXCEPTIONS,
                 allowlist: frozenset[str] | None = None) -> list[str]:
    """Return violation messages for the tree at ``root`` (empty list = clean)."""
    if allowlist is None:
        allowlist = PRE_MIGRATION_ALLOWLIST
    problems: list[str] = []

    for rel in sorted(_tests_files(root)):
        if DATED_RE.match(rel.rsplit("/", 1)[-1]) and rel not in exceptions and rel not in allowlist:
            problems.append(f"{rel}: date-named test file (name it test_<module>_<behaviour>.py; "
                            "put the date in the docstring)")

    tsv = root / RENAMES_REL
    if not tsv.is_file():
        problems.append(f"{RENAMES_REL}: missing")
        return problems
    seen_cols = False
    seen_old: set[str] = set()
    seen_new: set[str] = set()
    for n, line in enumerate(tsv.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        if not seen_cols:
            if line != COLUMN_LINE:
                problems.append(f"{RENAMES_REL}:{n}: column header line is not the expected one")
            seen_cols = True
            continue
        cells = line.split("\t")
        if len(cells) != len(COLUMNS):
            problems.append(f"{RENAMES_REL}:{n}: malformed row ({len(cells)} columns, expected {len(COLUMNS)})")
            continue
        row = dict(zip(COLUMNS, cells))
        kind, old, new = row["kind"], row["old_path"], row["new_path"]
        if old in seen_old:
            problems.append(f"{RENAMES_REL}:{n}: duplicate old_path {old}")
        seen_old.add(old)
        if kind == "exempt":
            if old not in exceptions:
                problems.append(f"{RENAMES_REL}:{n}: exempt row {old} is not in the approved exception set")
            if new != "-":
                problems.append(f"{RENAMES_REL}:{n}: exempt row must have new_path '-'")
            if row["reason"].strip() in ("", "-"):
                problems.append(f"{RENAMES_REL}:{n}: exempt row {old} has no reason")
            if not (root / old).is_file():
                problems.append(f"{RENAMES_REL}:{n}: stale exempt row, {old} no longer exists")
        elif kind == "rename":
            if new in seen_new:
                problems.append(f"{RENAMES_REL}:{n}: duplicate new_path {new}")
            seen_new.add(new)
            base = new.rsplit("/", 1)[-1]
            if not new.startswith("tests/") or not re.fullmatch(r"test_[A-Za-z0-9_]+\.py", base):
                problems.append(f"{RENAMES_REL}:{n}: new_path {new} is not a tests/**/test_*.py path")
            elif DATED_RE.match(base) and new not in exceptions:
                problems.append(f"{RENAMES_REL}:{n}: new_path {new} is date-named")
            if (root / old).exists():
                problems.append(f"{RENAMES_REL}:{n}: renamed row but old path {old} still exists")
            if not (root / new).is_file():
                problems.append(f"{RENAMES_REL}:{n}: renamed row but new path {new} is missing")
        else:
            problems.append(f"{RENAMES_REL}:{n}: unknown kind {kind!r}")
    if not seen_cols:
        problems.append(f"{RENAMES_REL}: no column header line")
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _tree(tmp_path: Path, files: dict[str, str], rows: list[str] | None = None) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    body = [COLUMN_LINE] + (rows or [])
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / RENAMES_REL).write_text("\n".join(body) + "\n", encoding="utf-8")
    return tmp_path


def _row(kind, old, new="-", reason="-"):
    return "\t".join([kind, old, new, "-", "-", "-", "-", "-", reason])


EXC = frozenset({"tests/unit/test_sep10_x.py"})


def test_real_tree_is_clean():
    assert check_naming(REPO_ROOT) == []


def test_approved_exception_set_is_exact():
    assert APPROVED_EXCEPTIONS == frozenset({"tests/unit/test_sep10_calendar_forced_action_loop.py"})


def test_allowlist_is_structurally_sound():
    for rel in PRE_MIGRATION_ALLOWLIST:
        assert rel.startswith("tests/") and DATED_RE.match(rel.rsplit("/", 1)[-1]), rel
        assert rel not in APPROVED_EXCEPTIONS, rel
    assert sorted(PRE_MIGRATION_ALLOWLIST) == sorted(set(PRE_MIGRATION_ALLOWLIST))


def test_clean_synthetic_tree(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_user_profile_supersession.py": "x=1\n"})
    assert check_naming(root, exceptions=EXC, allowlist=frozenset()) == []


def test_new_dated_file_is_rejected(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_oct20_new_thing.py": "x=1\n"})
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert len(msgs) == 1 and "date-named" in msgs[0]


def test_dated_name_is_case_insensitive_and_covers_other_dirs(tmp_path):
    root = _tree(tmp_path, {"tests/integration/test_SEP09_thing.py": "", "tests/test_Oct1a_x.py": ""})
    assert len(check_naming(root, exceptions=EXC, allowlist=frozenset())) == 2


def test_exception_and_allowlist_members_pass(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_sep10_x.py": "", "tests/unit/test_sep11_old.py": ""})
    assert check_naming(root, exceptions=EXC, allowlist=frozenset({"tests/unit/test_sep11_old.py"})) == []
    assert len(check_naming(root, exceptions=EXC, allowlist=frozenset())) == 1


def test_adding_a_tsv_row_cannot_raise_the_exception_set(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_sep12_sneaky.py": ""},
                 [_row("exempt", "tests/unit/test_sep12_sneaky.py", reason="because")])
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert any("not in the approved exception set" in m for m in msgs)
    assert any("date-named test file" in m for m in msgs)


def test_exempt_row_requirements(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_sep10_x.py": ""},
                 [_row("exempt", "tests/unit/test_sep10_x.py", "tests/unit/test_a.py", "-")])
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert any("new_path '-'" in m for m in msgs) and any("no reason" in m for m in msgs)


def test_valid_exempt_row_passes(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_sep10_x.py": ""},
                 [_row("exempt", "tests/unit/test_sep10_x.py", reason="class-guard ledger anchor")])
    assert check_naming(root, exceptions=EXC, allowlist=frozenset()) == []


def test_stale_exempt_row_is_rejected(tmp_path):
    root = _tree(tmp_path, {}, [_row("exempt", "tests/unit/test_sep10_x.py", reason="anchor")])
    assert any("stale exempt row" in m for m in check_naming(root, exceptions=EXC, allowlist=frozenset()))


def test_rename_row_old_must_be_gone_and_new_present(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_old_name.py": ""},
                 [_row("rename", "tests/unit/test_old_name.py", "tests/unit/test_new_name.py")])
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert any("still exists" in m for m in msgs) and any("is missing" in m for m in msgs)


def test_valid_rename_row_passes(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_user_profile_x.py": ""},
                 [_row("rename", "tests/unit/test_sep09_x.py", "tests/unit/test_user_profile_x.py")])
    assert check_naming(root, exceptions=EXC, allowlist=frozenset()) == []


def test_duplicate_paths_are_rejected(tmp_path):
    root = _tree(tmp_path, {"tests/unit/test_b.py": ""},
                 [_row("rename", "tests/unit/test_sep09_x.py", "tests/unit/test_b.py"),
                  _row("rename", "tests/unit/test_sep09_x.py", "tests/unit/test_b.py")])
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert any("duplicate old_path" in m for m in msgs) and any("duplicate new_path" in m for m in msgs)


def test_new_path_shape_is_enforced(tmp_path):
    root = _tree(tmp_path, {"scripts/test_b.py": "", "tests/unit/helper.py": "", "tests/unit/test_oct01_b.py": ""},
                 [_row("rename", "tests/unit/test_a1.py", "scripts/test_b.py"),
                  _row("rename", "tests/unit/test_a2.py", "tests/unit/helper.py"),
                  _row("rename", "tests/unit/test_a3.py", "tests/unit/test_oct01_b.py")])
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset({"tests/unit/test_oct01_b.py"}))
    assert sum("is not a tests/**/test_*.py path" in m for m in msgs) == 2
    assert any("is date-named" in m for m in msgs)


def test_malformed_rows_and_headers_are_rejected(tmp_path):
    root = _tree(tmp_path, {})
    (root / RENAMES_REL).write_text("kind\told_path\nrename\ta\n", encoding="utf-8")
    msgs = check_naming(root, exceptions=EXC, allowlist=frozenset())
    assert any("column header line" in m for m in msgs)
    (root / RENAMES_REL).write_text(COLUMN_LINE + "\nrename\ttoo\tfew\n", encoding="utf-8")
    assert any("malformed row" in m for m in check_naming(root, exceptions=EXC, allowlist=frozenset()))
    (root / RENAMES_REL).write_text(COLUMN_LINE + "\n" + _row("bogus", "a") + "\n", encoding="utf-8")
    assert any("unknown kind" in m for m in check_naming(root, exceptions=EXC, allowlist=frozenset()))


def test_missing_map_file_is_rejected(tmp_path):
    (tmp_path / "tests").mkdir()
    assert any("missing" in m for m in check_naming(tmp_path, exceptions=EXC, allowlist=frozenset()))


def test_header_comment_lines_are_ignored(tmp_path):
    root = _tree(tmp_path, {})
    (root / RENAMES_REL).write_text("#status\tapproved\n#base_sha\tabc\n" + COLUMN_LINE + "\n", encoding="utf-8")
    assert check_naming(root, exceptions=EXC, allowlist=frozenset()) == []
