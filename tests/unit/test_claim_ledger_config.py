"""claim_ledger config section (config/config.yaml + schema.py + app_config.py).

Incident 2026-10-20 (restatement drift). Classes: BC-75, BC-52, BC-82 (live
config). The committed defaults are ALL OFF, the schema validates them, a YAML
1.1 bare ``off`` for ``reply_check`` still means "off", and app_config exposes
the constants the live consumers read at call time.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import config.app_config as ac
from config.schema import ClaimLedgerSection, DaemonConfig

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["claim_ledger"]


def test_committed_section_validates_and_ships_entirely_off():
    section = ClaimLedgerSection(**COMMITTED)
    DaemonConfig(claim_ledger=COMMITTED)
    assert (section.enabled, section.render_block, section.reply_check) == (False, False, "off")
    assert COMMITTED["enabled"] is False and COMMITTED["render_block"] is False
    assert COMMITTED["reply_check"] == "off"


def test_committed_values_match_the_design():
    assert COMMITTED["model"] == "deepseek-v4.1-flash"
    assert (COMMITTED["write_timeout_s"], COMMITTED["max_block_lines"], COMMITTED["max_block_tokens"]) == (10.0, 12, 450)
    assert (COMMITTED["standing_window_days"], COMMITTED["deadline_window_days"]) == (30, 14)
    assert COMMITTED["store_path"] == os.path.join("data", "claim_ledger.json")


def test_app_config_constants_agree_with_the_committed_section():
    assert ac.CLAIM_LEDGER_ENABLED is False and ac.CLAIM_LEDGER_RENDER_BLOCK is False
    assert ac.CLAIM_LEDGER_REPLY_CHECK == "off"
    assert ac.CLAIM_LEDGER_MODEL == "deepseek-v4.1-flash"
    assert ac.CLAIM_LEDGER_WRITE_TIMEOUT_S == 10.0
    assert (ac.CLAIM_LEDGER_MAX_BLOCK_LINES, ac.CLAIM_LEDGER_MAX_BLOCK_TOKENS) == (12, 450)
    assert (ac.CLAIM_LEDGER_STANDING_WINDOW_DAYS, ac.CLAIM_LEDGER_DEADLINE_WINDOW_DAYS) == (30, 14)
    assert ac.CLAIM_LEDGER_STORE_PATH == os.path.join("data", "claim_ledger.json")


def test_a_bare_yaml_off_still_means_off():
    parsed = yaml.safe_load("reply_check: off")
    assert parsed["reply_check"] is False                       # the YAML 1.1 trap
    assert ClaimLedgerSection(**parsed).reply_check == "off"


@pytest.mark.parametrize("field,bad", [("reply_check", "correct"), ("write_timeout_s", 0),
                                       ("max_block_lines", 0), ("standing_window_days", 0)])
def test_schema_rejects_out_of_range_values(field, bad):
    with pytest.raises(ValidationError):
        ClaimLedgerSection(**{**COMMITTED, field: bad})


def test_log_only_is_accepted():
    assert ClaimLedgerSection(**{**COMMITTED, "reply_check": "log_only"}).reply_check == "log_only"


def test_conftest_sandbox_forces_the_switches_off_for_every_test():
    # tests/conftest.py::_sandbox_claim_ledger -- a config.local.yaml that enables the ledger must
    # never make a test spend credits.
    assert ac.CLAIM_LEDGER_ENABLED is False
