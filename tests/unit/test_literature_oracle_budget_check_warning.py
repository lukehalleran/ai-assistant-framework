"""literature_oracle.check_budget must WARN when the credit read fails.

Incident class: 2026-10-27 oct27b wave 3a swallow marking. class: BC-69, BC-70.
A failed credit read used to return 0.0 silently, so a driver stopped with no trace.
"""
import logging

from knowledge import literature_oracle


class _Boom:
    def get_remaining_credits(self):
        raise RuntimeError("limiter offline")


class _Ok:
    def get_remaining_credits(self):
        return 42


def test_failed_credit_read_warns_and_returns_zero(caplog):
    with caplog.at_level(logging.WARNING):
        assert literature_oracle.check_budget(_Boom()) == 0.0
    assert any("[LiteratureOracle] credit check failed" in r.getMessage()
               and r.levelno == logging.WARNING for r in caplog.records)
    assert not any("limiter offline" in r.getMessage() for r in caplog.records)


def test_healthy_credit_read_unchanged(caplog):
    with caplog.at_level(logging.WARNING):
        assert literature_oracle.check_budget(_Ok()) == 42.0
    assert not caplog.records
