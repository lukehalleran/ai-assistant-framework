"""
Phase 4: Generation harness for prompt ablation eval.

Runs all (snapshot, variant) combinations through side-effect-free LLM generation.
Produces a directory of EvalGenerationResult JSON files for downstream judging.

Flow:
    1. Load snapshots from disk
    2. Generate baseline (full prompt) for each snapshot
    3. Generate variants (LOO, AOI, bundle, reorder) for each snapshot
    4. Wrap each generation with PersistenceGuard
    5. Save results to eval/runs/<run_id>/
    6. Resume capability: skip already-generated pairs

Inputs:
    - eval/snapshots/*.json (from Phase 1 capture)
    - VariantGenerator + DEFAULT_BUNDLES (from Phase 2)
    - EvalGenerator (from Phase 1)
Outputs:
    - eval/runs/<run_id>/results/*.json — per-(snapshot, variant) generation results
    - eval/runs/<run_id>/manifest.json — run metadata and progress
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.no_store_generation import EvalGenerator
from eval.schema import EvalGenerationResult, PromptSnapshot
from eval.snapshots import SnapshotReplay, load_snapshot
from eval.variants import (
    DEFAULT_BUNDLES,
    PromptVariant,
    VariantGenerator,
    VariantStrategy,
)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    """Configuration for a generation run."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2048
    system_message: Optional[str] = None

    # Variant generation
    include_reorder: bool = False
    bundles: Optional[Dict[str, List[str]]] = None

    # Rate limiting
    requests_per_minute: int = 30
    delay_between_requests: float = 0.0  # seconds, overrides rpm if > 0

    # Persistence guard
    use_persistence_guard: bool = False  # requires chromadb client
    data_paths: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HarnessConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Generation pair tracking
# ---------------------------------------------------------------------------

BASELINE_VARIANT_ID = "__baseline__"


@dataclass
class GenerationPair:
    """Tracks one (snapshot, variant) generation pair."""

    snapshot_id: str
    variant_id: str  # BASELINE_VARIANT_ID for full prompt
    query_text: str
    strategy: str  # "baseline", "leave_one_out", etc.
    description: str
    sections_removed: List[str] = field(default_factory=list)
    sections_added: List[str] = field(default_factory=list)
    reordered_sections: Dict[str, int] = field(default_factory=dict)
    token_count_estimate: int = 0
    parent_token_count: int = 0

    def result_filename(self) -> str:
        """Filename for the result JSON."""
        return f"{self.snapshot_id}_{self.variant_id}.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Metadata for a generation run."""

    run_id: str
    config: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    total_pairs: int = 0
    completed_pairs: int = 0
    failed_pairs: int = 0
    skipped_pairs: int = 0  # already generated (resume)
    total_generation_time_ms: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    pairs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class GenerationHarness:
    """Batch generation orchestration for prompt ablation eval."""

    def __init__(
        self,
        generator: EvalGenerator,
        config: Optional[HarnessConfig] = None,
        persistence_guard: Any = None,
    ) -> None:
        self._generator = generator
        self._config = config or HarnessConfig()
        self._guard = persistence_guard
        self._replay = SnapshotReplay()
        self._variant_gen = VariantGenerator()

    def load_snapshots(self, snapshot_dir: str = "eval/snapshots") -> Dict[str, PromptSnapshot]:
        """Load all snapshot JSON files from a directory."""
        snapshots: Dict[str, PromptSnapshot] = {}
        snap_path = Path(snapshot_dir)
        if not snap_path.exists():
            return snapshots

        for path in sorted(snap_path.glob("*.json")):
            if path.name == ".gitignore":
                continue
            try:
                snapshot = load_snapshot(path)
                snapshots[snapshot.snapshot_id] = snapshot
            except Exception as e:
                print(f"  WARN: skipping {path.name}: {e}")

        return snapshots

    def plan_pairs(
        self,
        snapshots: Dict[str, PromptSnapshot],
    ) -> List[GenerationPair]:
        """Plan all (snapshot, variant) pairs to generate.

        For each snapshot:
        - One baseline (full prompt)
        - LOO, AOI, bundle, (optionally reorder) variants
        """
        bundles = self._config.bundles or DEFAULT_BUNDLES
        pairs: List[GenerationPair] = []

        for snap_id, snapshot in sorted(snapshots.items()):
            # Baseline
            layer = snapshot.layers.get("post_hygiene")
            if layer is None:
                continue

            all_keys = list(layer.sections.keys())
            baseline_tokens = sum(s.token_count for s in layer.sections.values())

            pairs.append(GenerationPair(
                snapshot_id=snap_id,
                variant_id=BASELINE_VARIANT_ID,
                query_text=snapshot.query_text,
                strategy="baseline",
                description="Full prompt (baseline)",
                token_count_estimate=baseline_tokens,
                parent_token_count=baseline_tokens,
            ))

            # Generate variants
            variants = self._variant_gen.generate_all(
                snapshot,
                bundles=bundles,
                include_reorder=self._config.include_reorder,
            )

            for v in variants:
                pairs.append(GenerationPair(
                    snapshot_id=snap_id,
                    variant_id=v.variant_id,
                    query_text=snapshot.query_text,
                    strategy=v.strategy.value,
                    description=v.description,
                    sections_removed=v.sections_removed,
                    sections_added=v.sections_added,
                    reordered_sections=v.reordered_sections,
                    token_count_estimate=v.token_count_estimate,
                    parent_token_count=v.parent_token_count,
                ))

        return pairs

    async def run(
        self,
        snapshots: Dict[str, PromptSnapshot],
        output_dir: str = "eval/runs",
        run_id: Optional[str] = None,
    ) -> RunManifest:
        """Execute all generation pairs.

        Args:
            snapshots: Map of snapshot_id -> PromptSnapshot.
            output_dir: Base directory for run output.
            run_id: Optional run ID. Auto-generated if None.

        Returns:
            RunManifest with results summary.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = Path(output_dir) / run_id
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"

        # Plan pairs
        pairs = self.plan_pairs(snapshots)

        manifest = RunManifest(
            run_id=run_id,
            config=self._config.to_dict(),
            started_at=datetime.now(timezone.utc).isoformat(),
            total_pairs=len(pairs),
        )

        # Check for existing results (resume)
        existing_results = set()
        if manifest_path.exists():
            try:
                old_manifest = RunManifest.load(manifest_path)
                for p in old_manifest.pairs:
                    if p.get("status") == "completed":
                        existing_results.add(p["result_filename"])
            except Exception:
                pass

        # Persistence guard snapshot (before)
        guard_before = None
        if self._guard is not None and self._config.use_persistence_guard:
            guard_before = self._guard.capture()

        # Rate limiting
        delay = self._config.delay_between_requests
        if delay <= 0 and self._config.requests_per_minute > 0:
            delay = 60.0 / self._config.requests_per_minute

        # Generate
        for i, pair in enumerate(pairs):
            result_file = results_dir / pair.result_filename()
            result_filename = pair.result_filename()

            # Resume: skip already completed
            if result_filename in existing_results:
                manifest.skipped_pairs += 1
                manifest.pairs.append({
                    **pair.to_dict(),
                    "status": "skipped",
                    "result_filename": result_filename,
                })
                continue

            # Replay prompt
            try:
                snapshot = snapshots[pair.snapshot_id]
                if pair.variant_id == BASELINE_VARIANT_ID:
                    prompt = self._replay.replay_from_layer(snapshot)
                else:
                    # Extract active sections from the variant
                    variant = self._find_variant(snapshot, pair.variant_id)
                    if variant is not None:
                        prompt = self._replay.replay_from_layer(
                            snapshot,
                            sections_to_include=variant.active_sections,
                        )
                    else:
                        # Fallback: baseline if variant not reproducible
                        prompt = self._replay.replay_from_layer(snapshot)
                        pair.description += " (variant not found, used baseline)"
            except Exception as e:
                manifest.failed_pairs += 1
                manifest.errors.append({
                    "pair": pair.to_dict(),
                    "error": str(e),
                    "stage": "replay",
                })
                manifest.pairs.append({
                    **pair.to_dict(),
                    "status": "failed",
                    "error": str(e),
                    "result_filename": result_filename,
                })
                continue

            # Generate
            try:
                result = await self._generator.generate(
                    assembled_prompt=prompt,
                    model=self._config.model,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    system_message=self._config.system_message,
                )

                # Save result
                result_data = {
                    "pair": pair.to_dict(),
                    "result": result.to_dict(),
                    "prompt_text": prompt,
                }
                result_file.write_text(json.dumps(result_data, indent=2))

                manifest.completed_pairs += 1
                manifest.total_generation_time_ms += result.generation_time_ms
                manifest.total_prompt_tokens += result.prompt_token_count
                manifest.total_response_tokens += result.response_token_count
                manifest.pairs.append({
                    **pair.to_dict(),
                    "status": "completed",
                    "result_filename": result_filename,
                    "generation_time_ms": result.generation_time_ms,
                    "prompt_tokens": result.prompt_token_count,
                    "response_tokens": result.response_token_count,
                })

            except Exception as e:
                manifest.failed_pairs += 1
                manifest.errors.append({
                    "pair": pair.to_dict(),
                    "error": str(e),
                    "stage": "generation",
                })
                manifest.pairs.append({
                    **pair.to_dict(),
                    "status": "failed",
                    "error": str(e),
                    "result_filename": result_filename,
                })

            # Save manifest after each pair (for resume)
            manifest.save(manifest_path)

            # Rate limiting delay (skip after last pair)
            if delay > 0 and i < len(pairs) - 1:
                await asyncio.sleep(delay)

        # Persistence guard check (after)
        if guard_before is not None and self._guard is not None:
            guard_after = self._guard.capture()
            try:
                guard_before.assert_same_as(guard_after)
            except AssertionError as e:
                manifest.errors.append({
                    "error": f"Persistence guard violation: {e}",
                    "stage": "persistence_guard",
                })

        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.save(manifest_path)

        return manifest

    def _find_variant(
        self,
        snapshot: PromptSnapshot,
        variant_id: str,
    ) -> Optional[PromptVariant]:
        """Regenerate a specific variant from a snapshot by ID."""
        bundles = self._config.bundles or DEFAULT_BUNDLES
        variants = self._variant_gen.generate_all(
            snapshot,
            bundles=bundles,
            include_reorder=self._config.include_reorder,
        )
        for v in variants:
            if v.variant_id == variant_id:
                return v
        return None


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_run_results(
    run_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Load all generation results from a run directory.

    Returns:
        Map of result_filename -> {pair, result, prompt_text}
    """
    results_path = Path(run_dir) / "results"
    results: Dict[str, Dict[str, Any]] = {}

    if not results_path.exists():
        return results

    for path in sorted(results_path.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            results[path.name] = data
        except Exception:
            continue

    return results


def load_run_manifest(run_dir: str) -> Optional[RunManifest]:
    """Load the manifest for a run."""
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    return RunManifest.load(manifest_path)
