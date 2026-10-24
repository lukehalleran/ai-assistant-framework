#!/usr/bin/env python3
"""Validate the DEPLOYED model capability table against OpenRouter reality.

This is the "test the fix itself" check: instead of re-deriving what a model can
do, it fetches OpenRouter's authoritative /models metadata and compares each
registered model's DECLARED capabilities (models.model_manager.MODEL_CAPABILITIES
— the same table the classifiers read) against the provider's own
`supported_parameters` and `input_modalities`.

It is what surfaced two live bugs on 2026-07-22: claude-fable-5 declared
vision/tools=False (OpenRouter: image input + tools=True) and deepseek-r1
declared tools=False (OpenRouter: tools=True).

Hard checks (exit non-zero on mismatch):
  * tools   <->  "tools" in supported_parameters
  * vision  <->  "image" in input_modalities

Advisory (warn only — our `reasoning` flag means "request reasoning SEPARATION",
which we may intentionally not do for some providers even when they can reason):
  * reasoning  vs  "reasoning"/"include_reasoning" in supported_parameters

No generation credits are spent — this only reads the public model catalog.
Requires network. Run:  python scripts/verify_model_capabilities_live.py
"""
import sys
import json
import urllib.request
from pathlib import Path

# Resolve `models` from THIS checkout, whatever the shell's PYTHONPATH points at
# (in a clone the login shell's path resolves the live repo's modules — BC-83).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_manager import API_MODEL_ALIASES, MODEL_CAPABILITIES, MODEL_CONTEXT_LIMITS  # noqa: E402

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_models():
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {m["id"]: m for m in data.get("data", [])}


def or_caps(model: dict):
    sp = set(model.get("supported_parameters", []) or [])
    arch = model.get("architecture", {}) or {}
    in_mods = arch.get("input_modalities") or []
    if not in_mods and arch.get("modality"):
        # older schema: "text->text" / "text+image->text"
        in_mods = arch["modality"].split("->")[0].split("+")
    return {
        "tools": "tools" in sp,
        "vision": "image" in in_mods,
        "reasoning": bool(sp & {"reasoning", "include_reasoning"}),
    }


def main() -> int:
    try:
        catalog = fetch_openrouter_models()
    except Exception as e:  # network / parse
        print(f"ERROR: could not fetch OpenRouter catalog: {e}")
        return 2

    slugs = sorted(set(API_MODEL_ALIASES.values()))
    hard_mismatches = []
    advisories = []
    missing = []

    print(f"{'model':<38} {'tools':>12} {'vision':>12} {'reasoning':>16}")
    print("-" * 80)
    for slug in slugs:
        declared = MODEL_CAPABILITIES.get(slug, {})
        model = catalog.get(slug)
        if model is None:
            missing.append(slug)
            print(f"{slug:<38} {'NOT ON OPENROUTER':>42}")
            continue
        actual = or_caps(model)

        def cell(dim):
            d, a = declared.get(dim), actual[dim]
            if d == a:
                mark = "ok"
            else:
                # reasoning is an advisory (separation policy), not a hard fact.
                mark = "adv" if dim == "reasoning" else "MISMATCH"
            return f"{str(d)}/{str(a)} {mark}"

        print(f"{slug:<38} {cell('tools'):>12} {cell('vision'):>12} {cell('reasoning'):>16}")

        for dim in ("tools", "vision"):
            if declared.get(dim) != actual[dim]:
                hard_mismatches.append((slug, dim, declared.get(dim), actual[dim]))
        # Advisories (not hard facts): a declared context length that disagrees with the
        # catalog, and sampling parameters the catalog does not list for a route whose row
        # does not declare sampling_params=False (only a request shows whether it refuses).
        ctx = model.get("context_length")
        if ctx and slug in MODEL_CONTEXT_LIMITS and MODEL_CONTEXT_LIMITS[slug] != ctx:
            advisories.append((slug, "context", MODEL_CONTEXT_LIMITS[slug], ctx))
        sp = set(model.get("supported_parameters", []) or [])
        if not ({"temperature", "top_p"} & sp) and declared.get("sampling_params") is not False:
            advisories.append((slug, "sampling_params", "unset", "catalog lists no temperature/top_p; run the probe"))
        if declared.get("reasoning") != actual["reasoning"]:
            advisories.append((slug, "reasoning", declared.get("reasoning"), actual["reasoning"]))

    print("-" * 80)
    if missing:
        print(f"\n⚠️  {len(missing)} registered model(s) not found on OpenRouter: {missing}")
    for slug, dim, d, a in advisories:
        note = {"reasoning": "ok if intentional — reasoning-separation policy",
                "context": "update MODEL_CONTEXT_LIMITS if the catalog is right",
                "sampling_params": "only scripts/probe_model_request_shapes.py --run can say"}[dim]
        print(f"  advisory: {slug} {dim} declared={d} but OpenRouter={a} ({note})")
    if hard_mismatches:
        print(f"\n❌ {len(hard_mismatches)} HARD mismatch(es) — declared caps disagree with OpenRouter:")
        for slug, dim, d, a in hard_mismatches:
            print(f"    {slug}: {dim} declared={d}, OpenRouter says {a}")
        return 1
    print("\n✅ All registered models' tools+vision match OpenRouter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
