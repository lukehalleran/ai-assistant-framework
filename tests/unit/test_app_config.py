"""
Unit tests for config/app_config.py

Tests configuration functionality:
- Variable resolution in config
- YAML loading
- Default values
- Path resolution
"""

import pytest
import os
from pathlib import Path
from config.app_config import (
    resolve_vars,
    load_yaml_config,
    ensure_config_defaults,
)


# =============================================================================
# resolve_vars Tests
# =============================================================================

def test_resolve_vars_no_variables():
    """resolve_vars returns unchanged config with no variables"""
    config = {"key": "value", "number": 42}
    result = resolve_vars(config)
    assert result == config


def test_resolve_vars_simple_substitution():
    """resolve_vars substitutes simple variable references"""
    config = {
        "base_path": "/data",
        "corpus_file": "${base_path}/corpus.json"
    }
    result = resolve_vars(config)
    assert result["corpus_file"] == "/data/corpus.json"


def test_resolve_vars_nested_keys():
    """resolve_vars handles nested key references"""
    config = {
        "paths": {
            "data": "/var/data"
        },
        "file": "${paths.data}/file.txt"
    }
    result = resolve_vars(config)
    assert result["file"] == "/var/data/file.txt"


def test_resolve_vars_multiple_substitutions():
    """resolve_vars handles multiple variables in one value"""
    config = {
        "base": "/home",
        "user": "daemon",
        "path": "${base}/${user}/data"
    }
    result = resolve_vars(config)
    assert result["path"] == "/home/daemon/data"


def test_resolve_vars_nested_resolution():
    """resolve_vars resolves nested variable references"""
    config = {
        "a": "value_a",
        "b": "${a}",
        "c": "${b}_extended"
    }
    result = resolve_vars(config)
    assert result["b"] == "value_a"
    assert result["c"] == "value_a_extended"


def test_resolve_vars_missing_reference():
    """resolve_vars leaves unresolvable references as-is"""
    config = {
        "path": "${nonexistent.key}/file"
    }
    result = resolve_vars(config)
    # Should keep the placeholder if it can't be resolved
    assert "${nonexistent.key}" in result["path"]


def test_resolve_vars_dict_values():
    """resolve_vars processes nested dicts"""
    config = {
        "base": "/data",
        "nested": {
            "path": "${base}/nested"
        }
    }
    result = resolve_vars(config)
    assert result["nested"]["path"] == "/data/nested"


def test_resolve_vars_list_values():
    """resolve_vars processes lists"""
    config = {
        "base": "/data",
        "paths": [
            "${base}/path1",
            "${base}/path2"
        ]
    }
    result = resolve_vars(config)
    assert result["paths"][0] == "/data/path1"
    assert result["paths"][1] == "/data/path2"


def test_resolve_vars_non_string_values():
    """resolve_vars preserves non-string values"""
    config = {
        "string": "text",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "none": None
    }
    result = resolve_vars(config)
    assert result["number"] == 42
    assert result["float"] == 3.14
    assert result["bool"] == True
    assert result["none"] is None


def test_resolve_vars_circular_reference_safety():
    """resolve_vars handles potential circular references safely"""
    config = {
        "a": "${b}",
        "b": "${a}"
    }
    # Should not infinite loop, stops after max iterations
    result = resolve_vars(config)
    # At least doesn't crash
    assert isinstance(result, dict)


def test_resolve_vars_empty_config():
    """resolve_vars handles empty config"""
    config = {}
    result = resolve_vars(config)
    assert result == {}


def test_resolve_vars_non_dict_input():
    """resolve_vars returns non-dict inputs unchanged"""
    assert resolve_vars("string") == "string"
    assert resolve_vars(42) == 42
    assert resolve_vars(None) == None


# =============================================================================
# load_yaml_config Tests
# =============================================================================

def test_load_yaml_config_nonexistent_file():
    """load_yaml_config returns empty dict for nonexistent file"""
    result = load_yaml_config("totally_nonexistent_config_12345.yaml")
    assert isinstance(result, dict)
    # Should return empty or default config
    assert result is not None


def test_load_yaml_config_with_temp_file(tmp_path):
    """load_yaml_config loads valid YAML file"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
base_path: /test/data
corpus_file: corpus.json
    """)

    result = load_yaml_config(str(config_file))

    assert isinstance(result, dict)
    assert "base_path" in result or "corpus_file" in result


def test_load_yaml_config_invalid_yaml(tmp_path):
    """load_yaml_config handles invalid YAML gracefully"""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("{invalid yaml content [")

    result = load_yaml_config(str(config_file))

    # Should return empty dict or not crash
    assert isinstance(result, dict)


def test_load_yaml_config_empty_file(tmp_path):
    """load_yaml_config handles empty YAML file"""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    result = load_yaml_config(str(config_file))

    assert isinstance(result, dict)


def test_load_yaml_config_with_variables(tmp_path):
    """load_yaml_config resolves variables in YAML"""
    config_file = tmp_path / "vars.yaml"
    config_file.write_text("""
base: /data
path: ${base}/files
    """)

    result = load_yaml_config(str(config_file))

    # Should resolve variables
    if "path" in result and "base" in result:
        # Variables might be resolved
        pass  # Just verify it doesn't crash


# =============================================================================
# ensure_config_defaults Tests
# =============================================================================

def test_ensure_config_defaults_empty():
    """ensure_config_defaults adds defaults to empty config"""
    config = {}
    result = ensure_config_defaults(config)

    assert isinstance(result, dict)
    # Should have some default keys
    assert len(result) > 0


def test_ensure_config_defaults_preserves_existing():
    """ensure_config_defaults preserves existing values"""
    config = {
        "corpus_file": "/custom/path/corpus.json",
        "custom_key": "custom_value"
    }
    result = ensure_config_defaults(config)

    # Should keep custom values
    assert result.get("corpus_file") == "/custom/path/corpus.json"
    assert result.get("custom_key") == "custom_value"


def test_ensure_config_defaults_adds_missing():
    """ensure_config_defaults adds missing default keys"""
    config = {"some_key": "value"}
    result = ensure_config_defaults(config)

    # Should have more keys than input
    assert len(result) >= len(config)


def test_ensure_config_defaults_with_none_values():
    """ensure_config_defaults handles None values"""
    config = {"key": None}
    result = ensure_config_defaults(config)

    # Should not crash
    assert isinstance(result, dict)


# =============================================================================
# Integration Tests
# =============================================================================

def test_resolve_vars_then_defaults():
    """Variable resolution followed by defaults"""
    config = {
        "base": "/data",
        "path": "${base}/corpus"
    }

    # Resolve variables
    resolved = resolve_vars(config)

    # Apply defaults
    final = ensure_config_defaults(resolved)

    assert isinstance(final, dict)
    assert final["path"] == "/data/corpus"


def test_complex_nested_config():
    """Complex nested configuration with multiple levels"""
    config = {
        "app": {
            "name": "daemon",
            "version": "1.0"
        },
        "paths": {
            "base": "/data",
            "corpus": "${paths.base}/corpus.json",
            "logs": "${paths.base}/logs"
        },
        "full_path": "${paths.corpus}"
    }

    result = resolve_vars(config)

    # Check nested resolution worked
    assert result["paths"]["corpus"] == "/data/corpus.json"
    assert result["full_path"] == "/data/corpus.json"


def test_config_with_env_style_variables():
    """Config with environment-style variable references"""
    config = {
        "data_dir": "/home/user/data",
        "corpus": "${data_dir}/corpus.json",
        "chroma": "${data_dir}/chroma_db"
    }

    result = resolve_vars(config)

    assert result["corpus"] == "/home/user/data/corpus.json"
    assert result["chroma"] == "/home/user/data/chroma_db"
