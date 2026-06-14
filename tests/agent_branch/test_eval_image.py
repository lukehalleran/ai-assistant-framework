# tests/agent_branch/test_eval_image.py
"""The curated 'light' eval image: deps curation + wiring (no podman/network).

Pins that the light image excludes the heavy ML/retrieval stack (so it stays
small/fast) while carrying enough to import Daemon's config / data-model / graph /
pure-logic modules, and that eval_image threads through run_branch + run_portfolio
so the PROOF can run in it while the worker image stays deps-free."""

import inspect
from pathlib import Path

from agent_branch import portfolio, provisioning, supervisor

_DIR = Path(provisioning.__file__).parent / "eval_image"

_HEAVY = ["torch", "transformers", "sentence-transformers", "chromadb", "faiss",
          "spacy", "scikit-learn", "open_clip", "gradio", "pandas", "pyarrow"]
_LIGHT = ["pydantic", "pyyaml", "networkx", "numpy", "python-dateutil", "orjson"]


def _req_lines() -> str:
    """Actual requirement lines (comments stripped — the header comment names the
    excluded heavy deps for humans, which must not trip the exclusion check)."""
    text = (_DIR / "requirements-light.txt").read_text(encoding="utf-8")
    return "\n".join(l.strip().lower() for l in text.splitlines()
                     if l.strip() and not l.strip().startswith("#"))


def test_light_requirements_excludes_heavy_ml():
    reqs = _req_lines()
    for heavy in _HEAVY:
        assert heavy not in reqs, f"heavy dep {heavy!r} leaked into the light image"


def test_light_requirements_includes_core_light_deps():
    reqs = _req_lines()
    for light in _LIGHT:
        assert light in reqs, f"expected light dep {light!r} missing"


def test_containerfile_present_and_pip_based():
    cf = (_DIR / "Containerfile.light").read_text(encoding="utf-8")
    assert "python:3.11-slim" in cf and "pip install" in cf


def test_eval_image_constant_and_builder_exist():
    assert provisioning.EVAL_IMAGE_LIGHT
    assert callable(provisioning.ensure_eval_image)


def test_eval_image_threads_through_run_branch_and_portfolio():
    assert "eval_image" in inspect.signature(supervisor.Supervisor.run_branch).parameters
    assert "eval_image" in inspect.signature(portfolio.run_portfolio).parameters
