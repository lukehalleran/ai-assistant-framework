"""Cheap guards for the isolated-runtime CI model and receipt contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model_cache_helper():
    return _load_helper("smoke_model_cache_helper", "scripts/prepare_smoke_model_cache.py")


@pytest.fixture(scope="module")
def junit_helper():
    return _load_helper("smoke_junit_helper", "scripts/verify_smoke_junit.py")


def test_model_discovery_uses_valid_hub_repositories_and_tokenizer_fallback(model_cache_helper):
    requirements = model_cache_helper.discover_model_requirements(REPO_ROOT)

    assert set(requirements) == {
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "gpt2",
    }
    assert requirements["gpt2"] == {"tokenizer"}
    assert requirements["sentence-transformers/all-MiniLM-L6-v2"] == {"sentence_transformer"}
    assert requirements["BAAI/bge-small-en-v1.5"] == {"sentence_transformer"}
    assert requirements["cross-encoder/ms-marco-MiniLM-L-6-v2"] == {"cross_encoder"}


def test_gpt2_provisioning_is_tokenizer_only(model_cache_helper):
    patterns = model_cache_helper.allow_patterns_for("gpt2", {"tokenizer"})

    assert {"vocab.json", "merges.txt"} <= set(patterns)
    assert not any(pattern.endswith((".bin", ".safetensors", ".pt", ".pth")) for pattern in patterns)


def test_missing_gpt2_tokenizer_files_fail_provision_validation(model_cache_helper, tmp_path):
    with pytest.raises(RuntimeError, match="missing files"):
        model_cache_helper.validate_snapshot("gpt2", {"tokenizer"}, tmp_path)


def test_provisioning_passes_exact_repo_ids_and_checks_artifacts(model_cache_helper, tmp_path):
    calls = []

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns))
        target = tmp_path / repo_id.replace("/", "__")
        target.mkdir(parents=True)
        if repo_id == "gpt2":
            (target / "vocab.json").write_text("{}")
            (target / "merges.txt").write_text("#version: 0.2")
        else:
            (target / "config.json").write_text("{}")
            (target / "model.safetensors").write_bytes(b"stub")
        return str(target)

    receipt = model_cache_helper.provision_models(fake_snapshot_download, REPO_ROOT)

    assert {model_id for model_id, _ in calls} == {
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "gpt2",
    }
    gpt2_patterns = dict(calls)["gpt2"]
    assert "vocab.json" in gpt2_patterns and "merges.txt" in gpt2_patterns
    assert not any(pattern.endswith((".bin", ".safetensors", ".pt", ".pth")) for pattern in gpt2_patterns)
    assert len(receipt) == 4


def _junit(case_xml: str = "") -> str:
    return f"<testsuites><testsuite>{case_xml}</testsuite></testsuites>"


def _case(name: str, classname: str = "tests.smoke.test_fresh_clone_contract", child: str = "") -> str:
    return f'<testcase classname="{classname}" name="{name}">{child}</testcase>'


def test_junit_receipt_accepts_only_the_named_passing_two_boot_test(junit_helper, tmp_path):
    receipt = tmp_path / "smoke.xml"
    receipt.write_text(_junit(_case("test_isolated_runtime_two_boot_cycle")))

    assert junit_helper.verify(receipt) == (True, "JUnit receipt verified: 1 passed, 0 skipped")


@pytest.mark.parametrize(
    "contents",
    [
        _junit(),
        _junit(_case("test_unrelated")),
        _junit(_case("test_isolated_runtime_two_boot_cycle", child="<skipped/>")),
        _junit(_case("test_isolated_runtime_two_boot_cycle", child="<failure/>")),
        _junit(
            _case("test_isolated_runtime_two_boot_cycle")
            + _case("test_unrelated")
        ),
    ],
    ids=["empty", "wrong-test", "skipped", "failed", "extra-case"],
)
def test_junit_receipt_rejects_empty_skipped_failed_or_wrong_tests(junit_helper, tmp_path, contents):
    receipt = tmp_path / "smoke.xml"
    receipt.write_text(contents)

    valid, _ = junit_helper.verify(receipt)
    assert valid is False


def test_workflow_contract_detects_removing_the_smoke_step(junit_helper):
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert junit_helper.workflow_problems(workflow) == []

    mutated = workflow.replace(
        "tests/smoke/test_fresh_clone_contract.py", "tests/smoke/removed.py"
    )
    problems = junit_helper.workflow_problems(mutated)
    assert any("two-boot test" in problem for problem in problems)
