#!/usr/bin/env python3
"""Provision the exact local models constructed by the smoke path.

Model IDs are read from the deployed constructor calls rather than copied into
CI YAML. This keeps the offline smoke provisioning aligned with the source.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCES = (
    Path("models/model_manager.py"),
    Path("models/tokenizer_manager.py"),
    Path("memory/storage/multi_collection_chroma_store.py"),
    Path("memory/memory_retriever.py"),
)
MODEL_CONSTRUCTORS = {
    "SentenceTransformer": "sentence_transformer",
    "SentenceTransformerEmbeddingFunction": "sentence_transformer",
    "CrossEncoder": "cross_encoder",
}
TOKENIZER_CONSTRUCTOR = "AutoTokenizer"
_MODEL_WEIGHT_PATTERNS = (
    "*.json", "**/*.json", "*.txt", "**/*.txt", "*.model", "**/*.model",
    "*.safetensors", "**/*.safetensors", "*.bin", "**/*.bin",
    "*.pt", "**/*.pt", "*.pth", "**/*.pth", "*.onnx", "**/*.onnx",
)
_TOKENIZER_ONLY_PATTERNS = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
)


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _string_arg(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _canonical_sentence_transformer_id(model_id: str) -> str:
    """SentenceTransformers accepts short catalog names; Hub requires owner/repo."""
    if "/" not in model_id:
        return f"sentence-transformers/{model_id}"
    return model_id


def discover_model_requirements(root: Path | None = None) -> dict[str, set[str]]:
    """Return Hub repo IDs and roles read from the smoke path's constructors."""
    root = root or ROOT
    requirements: dict[str, set[str]] = {}
    for source_rel in MODEL_SOURCES:
        source = root / source_rel
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in MODEL_CONSTRUCTORS and node.args:
                model_id = _string_arg(node.args[0])
                if model_id:
                    role = MODEL_CONSTRUCTORS[name]
                    if role == "sentence_transformer":
                        model_id = _canonical_sentence_transformer_id(model_id)
                    requirements.setdefault(model_id, set()).add(role)
            # TokenizerManager's explicit HF fallback is exercised when the
            # configured local/API model tokenizer cannot be loaded.
            if name == "from_pretrained" and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id == TOKENIZER_CONSTRUCTOR and node.args:
                    model_id = _string_arg(node.args[0])
                    if model_id:
                        requirements.setdefault(model_id, set()).add("tokenizer")
            # The Chroma embedding constructor reads CHROMA_ST_MODEL from the
            # environment and uses its literal source default otherwise.
            if name == "getenv" and node.args and _string_arg(node.args[0]) == "CHROMA_ST_MODEL":
                if len(node.args) > 1:
                    model_id = _string_arg(node.args[1])
                    if model_id:
                        requirements.setdefault(model_id, set()).add("sentence_transformer")
    if not requirements:
        raise RuntimeError("No model IDs found in the smoke path constructor sources")
    return requirements


def allow_patterns_for(model_id: str, roles: set[str]) -> list[str]:
    if roles == {"tokenizer"}:
        return list(_TOKENIZER_ONLY_PATTERNS)
    return list(_MODEL_WEIGHT_PATTERNS)


def validate_snapshot(model_id: str, roles: set[str], snapshot_dir: Path) -> None:
    """Fail before CI smoke if the selected cache snapshot is incomplete."""
    files = {path.name for path in snapshot_dir.rglob("*") if path.is_file()}
    if "tokenizer" in roles and model_id == "gpt2":
        missing = {"vocab.json", "merges.txt"} - files
        if missing:
            raise RuntimeError(
                f"Tokenizer snapshot {model_id!r} is missing files: {sorted(missing)}"
            )
    if roles & {"sentence_transformer", "cross_encoder"}:
        has_config = "config.json" in files
        has_weights = any(
            name.endswith((".safetensors", ".bin", ".pt", ".pth", ".onnx"))
            for name in files
        )
        if not has_config or not has_weights:
            raise RuntimeError(
                f"Model snapshot {model_id!r} is incomplete: "
                f"config={has_config}, weights={has_weights}"
            )


def provision_models(snapshot_download, root: Path | None = None) -> list[dict]:
    entries = []
    for model_id, roles in sorted(discover_model_requirements(root).items()):
        path = Path(snapshot_download(
            repo_id=model_id,
            allow_patterns=allow_patterns_for(model_id, roles),
        ))
        validate_snapshot(model_id, roles, path)
        entries.append({"model_id": model_id, "roles": sorted(roles), "snapshot_path": str(path)})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    entries = provision_models(snapshot_download)

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps({"models": entries}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Provisioned {len(entries)} model/tokenizer snapshots from smoke-path constructors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
