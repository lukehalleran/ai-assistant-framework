"""
Phase 5: Pairwise judge harness for prompt ablation eval.

Compares baseline response (full prompt) against each variant response
using LLM-as-judge with structured rubrics. Position randomization
prevents ordering bias. Blind judging — judge doesn't know which is
baseline vs variant.

Inputs:
    - Generation results from Phase 4 (eval/runs/<run_id>/results/)
    - Judge model configuration
Outputs:
    - JudgeVerdict per (baseline, variant) pair
    - Aggregate win/loss/tie counts per section and strategy
    - JSON report at eval/runs/<run_id>/judgments/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.no_store_generation import EvalGenerator


# ---------------------------------------------------------------------------
# Rubric and verdict models
# ---------------------------------------------------------------------------

JUDGE_CRITERIA = [
    "accuracy",       # Factual correctness given the context provided
    "helpfulness",    # How well the response addresses the user's intent
    "conciseness",    # Appropriate length — not padded, not truncated
    "tone",           # Natural, matches conversational register
    "grounding",      # References context appropriately, no confabulation
]


@dataclass
class JudgeVerdict:
    """Result of judging one (baseline, variant) pair."""

    snapshot_id: str
    variant_id: str
    strategy: str
    sections_removed: List[str]
    sections_added: List[str]
    query_text: str

    # Core verdict
    winner: str  # "A", "B", "tie"
    winner_is_baseline: Optional[bool] = None  # resolved after unblinding
    confidence: float = 0.0  # 0-1
    explanation: str = ""

    # Per-criterion scores (1-5 each, for both A and B)
    scores_a: Dict[str, int] = field(default_factory=dict)
    scores_b: Dict[str, int] = field(default_factory=dict)

    # Position tracking
    baseline_position: str = ""  # "A" or "B" (which position baseline was in)

    # Metadata
    judge_model: str = ""
    judge_time_ms: int = 0
    timestamp: str = ""
    raw_judge_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> JudgeVerdict:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating conversational AI responses. You will be shown a user query and two responses (A and B). You do not know which response used the full prompt and which had sections removed.

Score each response on these criteria (1-5 scale):
- accuracy: Factual correctness given the context the model had
- helpfulness: How well it addresses what the user actually wants
- conciseness: Appropriate length — brief for casual, thorough for complex. Penalize padding/filler.
- tone: Natural conversational register, matches the user's energy
- grounding: References real context, doesn't make up facts or apps or names

Then pick a winner: A, B, or tie.

IMPORTANT: A shorter response that directly answers the query is BETTER than a longer one that pads with generic advice. Filler like "feel free to ask!" or unsolicited suggestions are negatives, not positives."""


def _build_judge_prompt(
    query: str,
    response_a: str,
    response_b: str,
) -> str:
    return f"""## User Query
{query}

## Response A
{response_a}

## Response B
{response_b}

## Your Evaluation
For each response, score these criteria from 1 (worst) to 5 (best):
- accuracy
- helpfulness
- conciseness
- tone
- grounding

Then state your verdict.

Respond in this exact format:
SCORES_A: accuracy=N helpfulness=N conciseness=N tone=N grounding=N
SCORES_B: accuracy=N helpfulness=N conciseness=N tone=N grounding=N
WINNER: A/B/tie
CONFIDENCE: 0.N
EXPLANATION: One sentence explaining your choice."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse the structured judge output into a dict.

    Returns dict with keys: scores_a, scores_b, winner, confidence, explanation.
    Returns partial results on parse failure rather than raising.
    """
    result: Dict[str, Any] = {
        "scores_a": {},
        "scores_b": {},
        "winner": "tie",
        "confidence": 0.5,
        "explanation": "",
    }

    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("SCORES_A:"):
            result["scores_a"] = _parse_scores(line[len("SCORES_A:"):])
        elif line.startswith("SCORES_B:"):
            result["scores_b"] = _parse_scores(line[len("SCORES_B:"):])
        elif line.startswith("WINNER:"):
            w = line[len("WINNER:"):].strip().upper()
            if w in ("A", "B", "TIE"):
                result["winner"] = w.lower() if w == "TIE" else w
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line[len("CONFIDENCE:"):].strip())
            except ValueError:
                pass
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line[len("EXPLANATION:"):].strip()

    return result


def _parse_scores(score_str: str) -> Dict[str, int]:
    """Parse 'accuracy=4 helpfulness=3 ...' into dict."""
    scores: Dict[str, int] = {}
    for part in score_str.strip().split():
        if "=" in part:
            key, _, val = part.partition("=")
            key = key.strip().lower()
            try:
                scores[key] = int(val.strip())
            except ValueError:
                pass
    return scores


# ---------------------------------------------------------------------------
# Pairwise judge
# ---------------------------------------------------------------------------

class PairwiseJudge:
    """Judges baseline vs variant response pairs using LLM-as-judge."""

    def __init__(
        self,
        generator: EvalGenerator,
        judge_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self._generator = generator
        self._judge_model = judge_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._rng = random.Random(seed)

    def _randomize_position(
        self,
        baseline_response: str,
        variant_response: str,
        snapshot_id: str,
        variant_id: str,
    ) -> Tuple[str, str, str]:
        """Randomly assign baseline/variant to A/B positions.

        Uses a deterministic seed per pair so results are reproducible.

        Returns:
            (response_a, response_b, baseline_position)
        """
        # Deterministic per-pair seed
        pair_key = f"{snapshot_id}:{variant_id}"
        pair_hash = int(hashlib.md5(pair_key.encode()).hexdigest()[:8], 16)
        if pair_hash % 2 == 0:
            return baseline_response, variant_response, "A"
        else:
            return variant_response, baseline_response, "B"

    async def judge_pair(
        self,
        query: str,
        baseline_response: str,
        variant_response: str,
        snapshot_id: str,
        variant_id: str,
        strategy: str,
        sections_removed: List[str],
        sections_added: List[str],
    ) -> JudgeVerdict:
        """Judge a single baseline vs variant pair.

        Randomizes A/B position, calls LLM judge, parses verdict,
        then unblinds to record which was baseline.
        """
        response_a, response_b, baseline_pos = self._randomize_position(
            baseline_response, variant_response, snapshot_id, variant_id,
        )

        prompt = _build_judge_prompt(query, response_a, response_b)

        t_start = time.perf_counter()
        raw_output = await self._generator.generate(
            assembled_prompt=prompt,
            model=self._judge_model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            system_message=JUDGE_SYSTEM_PROMPT,
        )
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        parsed = parse_judge_response(raw_output.response_text)

        # Unblind: determine if baseline won
        winner = parsed["winner"]
        if winner in ("A", "B"):
            winner_is_baseline = (winner == baseline_pos)
        else:
            winner_is_baseline = None  # tie

        return JudgeVerdict(
            snapshot_id=snapshot_id,
            variant_id=variant_id,
            strategy=strategy,
            sections_removed=sections_removed,
            sections_added=sections_added,
            query_text=query,
            winner=winner,
            winner_is_baseline=winner_is_baseline,
            confidence=parsed["confidence"],
            explanation=parsed["explanation"],
            scores_a=parsed["scores_a"],
            scores_b=parsed["scores_b"],
            baseline_position=baseline_pos,
            judge_model=self._judge_model,
            judge_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_judge_output=raw_output.response_text,
        )


# ---------------------------------------------------------------------------
# Batch judge runner
# ---------------------------------------------------------------------------

@dataclass
class JudgeRunManifest:
    """Tracks progress of a judging run."""

    run_id: str
    source_generation_run: str
    judge_model: str
    started_at: str
    completed_at: Optional[str] = None
    total_pairs: int = 0
    completed_pairs: int = 0
    failed_pairs: int = 0
    skipped_pairs: int = 0
    baseline_wins: int = 0
    variant_wins: int = 0
    ties: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> JudgeRunManifest:
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BatchJudge:
    """Runs pairwise judging across all variant results from a Phase 4 run."""

    def __init__(
        self,
        judge: PairwiseJudge,
        requests_per_minute: int = 30,
    ) -> None:
        self._judge = judge
        self._rpm = requests_per_minute

    def _load_generation_results(
        self,
        gen_run_dir: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Load Phase 4 results grouped by snapshot_id.

        Returns: {snapshot_id: {variant_id: result_data}}
        """
        results_path = Path(gen_run_dir) / "results"
        grouped: Dict[str, Dict[str, Any]] = {}

        for path in sorted(results_path.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                snap_id = data["pair"]["snapshot_id"]
                vid = data["pair"]["variant_id"]
                if snap_id not in grouped:
                    grouped[snap_id] = {}
                grouped[snap_id][vid] = data
            except Exception:
                continue

        return grouped

    def plan_judgments(
        self,
        gen_run_dir: str,
    ) -> List[Dict[str, Any]]:
        """Plan all judgment pairs from a generation run.

        For each snapshot, pairs the baseline with every variant.
        """
        grouped = self._load_generation_results(gen_run_dir)
        pairs: List[Dict[str, Any]] = []

        for snap_id, variants in sorted(grouped.items()):
            baseline = variants.get("__baseline__")
            if baseline is None:
                continue

            for vid, data in sorted(variants.items()):
                if vid == "__baseline__":
                    continue
                pairs.append({
                    "snapshot_id": snap_id,
                    "variant_id": vid,
                    "query_text": baseline["pair"]["query_text"],
                    "baseline_response": baseline["result"]["response_text"],
                    "variant_response": data["result"]["response_text"],
                    "strategy": data["pair"]["strategy"],
                    "sections_removed": data["pair"].get("sections_removed", []),
                    "sections_added": data["pair"].get("sections_added", []),
                })

        return pairs

    async def run(
        self,
        gen_run_dir: str,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> JudgeRunManifest:
        """Run judging across all baseline/variant pairs.

        Args:
            gen_run_dir: Path to Phase 4 run directory.
            output_dir: Where to save judgments. Defaults to gen_run_dir/judgments/.
            run_id: Optional run ID for resume.
        """
        if output_dir is None:
            output_dir = str(Path(gen_run_dir) / "judgments")

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        judge_dir = Path(output_dir) / run_id
        results_dir = judge_dir / "verdicts"
        results_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = judge_dir / "manifest.json"

        pairs = self.plan_judgments(gen_run_dir)

        # Get generation run id from dir name
        gen_run_id = Path(gen_run_dir).name

        manifest = JudgeRunManifest(
            run_id=run_id,
            source_generation_run=gen_run_id,
            judge_model=self._judge._judge_model,
            started_at=datetime.now(timezone.utc).isoformat(),
            total_pairs=len(pairs),
        )

        # Check for existing verdicts (resume)
        existing = set()
        if manifest_path.exists():
            try:
                old = JudgeRunManifest.load(manifest_path)
                for f in results_dir.glob("*.json"):
                    existing.add(f.stem)
            except Exception:
                pass

        delay = 60.0 / self._rpm if self._rpm > 0 else 0

        for i, pair_info in enumerate(pairs):
            verdict_key = f"{pair_info['snapshot_id']}_{pair_info['variant_id']}"
            verdict_file = results_dir / f"{verdict_key}.json"

            if verdict_key in existing:
                manifest.skipped_pairs += 1
                continue

            try:
                verdict = await self._judge.judge_pair(
                    query=pair_info["query_text"],
                    baseline_response=pair_info["baseline_response"],
                    variant_response=pair_info["variant_response"],
                    snapshot_id=pair_info["snapshot_id"],
                    variant_id=pair_info["variant_id"],
                    strategy=pair_info["strategy"],
                    sections_removed=pair_info["sections_removed"],
                    sections_added=pair_info["sections_added"],
                )

                verdict_file.write_text(json.dumps(verdict.to_dict(), indent=2))
                manifest.completed_pairs += 1

                if verdict.winner_is_baseline is True:
                    manifest.baseline_wins += 1
                elif verdict.winner_is_baseline is False:
                    manifest.variant_wins += 1
                else:
                    manifest.ties += 1

            except Exception as e:
                manifest.failed_pairs += 1
                manifest.errors.append({
                    "snapshot_id": pair_info["snapshot_id"],
                    "variant_id": pair_info["variant_id"],
                    "error": str(e),
                })

            manifest.save(manifest_path)

            if delay > 0 and i < len(pairs) - 1:
                await asyncio.sleep(delay)

        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.save(manifest_path)
        return manifest


# ---------------------------------------------------------------------------
# Result loading and aggregation
# ---------------------------------------------------------------------------

def load_verdicts(judge_dir: str) -> List[JudgeVerdict]:
    """Load all verdicts from a judge run directory."""
    verdicts_path = Path(judge_dir) / "verdicts"
    verdicts: List[JudgeVerdict] = []
    if not verdicts_path.exists():
        return verdicts
    for path in sorted(verdicts_path.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            verdicts.append(JudgeVerdict.from_dict(data))
        except Exception:
            continue
    return verdicts


def aggregate_by_section(
    verdicts: List[JudgeVerdict],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate LOO verdicts by removed section.

    Returns: {section: {baseline_wins, variant_wins, ties, total,
                         baseline_win_rate, avg_confidence}}
    """
    from collections import defaultdict

    section_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "baseline_wins": 0,
            "variant_wins": 0,
            "ties": 0,
            "total": 0,
            "confidences": [],
            "score_deltas": {c: [] for c in JUDGE_CRITERIA},
        }
    )

    for v in verdicts:
        if v.strategy != "leave_one_out" or not v.sections_removed:
            continue
        section = v.sections_removed[0]
        stats = section_stats[section]
        stats["total"] += 1

        if v.winner_is_baseline is True:
            stats["baseline_wins"] += 1
        elif v.winner_is_baseline is False:
            stats["variant_wins"] += 1
        else:
            stats["ties"] += 1

        stats["confidences"].append(v.confidence)

        # Score deltas (baseline score - variant score)
        for criterion in JUDGE_CRITERIA:
            bl_pos = v.baseline_position
            var_pos = "B" if bl_pos == "A" else "A"
            bl_scores = v.scores_a if bl_pos == "A" else v.scores_b
            var_scores = v.scores_a if var_pos == "A" else v.scores_b
            if criterion in bl_scores and criterion in var_scores:
                delta = bl_scores[criterion] - var_scores[criterion]
                stats["score_deltas"][criterion].append(delta)

    # Compute averages
    result: Dict[str, Dict[str, Any]] = {}
    for section, stats in sorted(section_stats.items()):
        total = stats["total"]
        confs = stats["confidences"]
        result[section] = {
            "baseline_wins": stats["baseline_wins"],
            "variant_wins": stats["variant_wins"],
            "ties": stats["ties"],
            "total": total,
            "baseline_win_rate": stats["baseline_wins"] / total if total > 0 else 0,
            "variant_win_rate": stats["variant_wins"] / total if total > 0 else 0,
            "avg_confidence": sum(confs) / len(confs) if confs else 0,
            "avg_score_deltas": {
                c: sum(deltas) / len(deltas) if deltas else 0
                for c, deltas in stats["score_deltas"].items()
            },
        }

    return result


def format_section_report(section_stats: Dict[str, Dict[str, Any]]) -> str:
    """Human-readable report of per-section LOO judging results."""
    lines = [
        "=== Section Ablation Judge Report (LOO) ===",
        "",
        "Baseline win = removing section HURT quality (section is valuable)",
        "Variant win  = removing section HELPED quality (section is harmful)",
        "",
        f"{'Section removed':<28s} {'BL wins':>8s} {'Var wins':>9s} {'Ties':>5s} "
        f"{'BL win%':>8s} {'Conf':>5s} {'N':>4s}",
        "-" * 72,
    ]

    # Sort by baseline win rate descending (most valuable sections first)
    for section, stats in sorted(
        section_stats.items(),
        key=lambda x: x[1]["baseline_win_rate"],
        reverse=True,
    ):
        lines.append(
            f"{section:<28s} {stats['baseline_wins']:>8d} "
            f"{stats['variant_wins']:>9d} {stats['ties']:>5d} "
            f"{stats['baseline_win_rate']:>7.0%} "
            f"{stats['avg_confidence']:>5.2f} "
            f"{stats['total']:>4d}"
        )

    # Per-criterion deltas
    lines.append("")
    lines.append("--- Per-Criterion Score Deltas (baseline - variant, positive = section helps) ---")
    lines.append(
        f"{'Section':<28s} {'accuracy':>9s} {'helpful':>8s} {'concise':>8s} "
        f"{'tone':>5s} {'ground':>7s}"
    )
    lines.append("-" * 70)
    for section, stats in sorted(
        section_stats.items(),
        key=lambda x: x[1]["baseline_win_rate"],
        reverse=True,
    ):
        deltas = stats["avg_score_deltas"]
        lines.append(
            f"{section:<28s} "
            f"{deltas.get('accuracy', 0):>+8.2f} "
            f"{deltas.get('helpfulness', 0):>+8.2f} "
            f"{deltas.get('conciseness', 0):>+8.2f} "
            f"{deltas.get('tone', 0):>+5.2f} "
            f"{deltas.get('grounding', 0):>+7.2f}"
        )

    return "\n".join(lines)
