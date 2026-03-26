# knowledge/implementation_detector.py
"""
Automatic detection of whether pending code proposals have been implemented.

Module Contract
- Purpose: 4-stage pipeline to check if a CodeProposal's changes already exist
  in the codebase. Stages: file existence → code content grep → git history →
  LLM judgment (borderline only).
- Inputs: CodeProposal objects with affected_files, implementation_steps, title
- Outputs: DetectionResult per proposal with confidence score and evidence
- Key behaviors:
  - Cooldown: skips proposals checked within IMPL_TRACKING_COOLDOWN seconds
  - Lightweight mode: file existence only (for shutdown hooks)
  - LLM stage only fires for borderline cases (0.30-0.84 confidence)
  - Never modifies proposal status directly (caller decides)
- Side effects: subprocess calls for git log; optional LLM call
- Dependencies: config.app_config, memory.code_proposal, models.model_manager (optional)
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("implementation_detector")


class DetectionResult(BaseModel):
    """Result of checking whether a proposal has been implemented."""

    proposal_id: str = ""
    confidence: float = 0.0
    status: str = "not_checked"
    evidence: str = ""
    file_existence_ratio: float = 0.0
    code_match_ratio: float = 0.0
    git_match_score: float = 0.0
    llm_adjusted: bool = False
    skipped_reason: str = ""


class ImplementationDetector:
    """
    Detects whether code proposals have been implemented in the codebase.

    4-stage pipeline:
    1. File existence — do the affected files exist?
    2. Code content — do identifiers from implementation steps appear in files?
    3. Git history — do recent commits touch affected files with relevant keywords?
    4. LLM judgment — for borderline cases, ask the model to adjudicate
    """

    def __init__(self, repo_path: str = ".", git_extractor=None, model_manager=None):
        self.repo_path = Path(repo_path)
        self.git_extractor = git_extractor
        self.model_manager = model_manager

    def detect_single(self, proposal, lightweight: bool = False) -> DetectionResult:
        """
        Run detection pipeline for a single proposal.

        Args:
            proposal: CodeProposal to check
            lightweight: If True, only run Stage 1 (file existence)

        Returns:
            DetectionResult with confidence and evidence
        """
        result = DetectionResult(proposal_id=proposal.id)

        # Check cooldown
        if self._should_skip_cooldown(proposal):
            result.skipped_reason = "cooldown"
            result.confidence = proposal.implementation_confidence
            result.status = proposal.implementation_status
            result.evidence = proposal.implementation_evidence
            return result

        # Stage 1: File existence
        existing_files = []
        file_ratio = self._stage_file_existence(proposal, existing_files)
        result.file_existence_ratio = file_ratio

        if lightweight:
            result.confidence = file_ratio
            result.status = self._confidence_to_status(result.confidence)
            result.evidence = self._build_evidence(
                file_ratio, [], [], llm_adjusted=False
            )
            return result

        # Stage 2: Code content
        code_ratio, code_matches = self._stage_code_content(proposal, existing_files)
        result.code_match_ratio = code_ratio

        # Stage 3: Git history
        git_score, git_matches = self._stage_git_history(proposal)
        result.git_match_score = git_score

        # Compute composite confidence
        result.confidence = self._compute_confidence(file_ratio, code_ratio, git_score)
        result.status = self._confidence_to_status(result.confidence)
        result.evidence = self._build_evidence(
            file_ratio, code_matches, git_matches, llm_adjusted=False
        )

        return result

    async def detect_batch(
        self, proposals: list, lightweight: bool = False
    ) -> List[DetectionResult]:
        """
        Run detection for a batch of proposals.

        Stages 1-3 run synchronously per proposal. Borderline results
        (0.30-0.84 confidence) are batched for a single LLM call.
        """
        results = []
        borderline = []

        for proposal in proposals:
            result = self.detect_single(proposal, lightweight=lightweight)
            results.append(result)

            if (
                not lightweight
                and not result.skipped_reason
                and 0.30 <= result.confidence < 0.84
            ):
                borderline.append((proposal, result))

        # Stage 4: LLM judgment for borderline cases
        if borderline and not lightweight and self.model_manager:
            adjustments = await self._stage_llm_judgment(borderline)
            for proposal, result in borderline:
                if proposal.id in adjustments:
                    result.confidence = adjustments[proposal.id]
                    result.status = self._confidence_to_status(result.confidence)
                    result.llm_adjusted = True
                    result.evidence = self._build_evidence(
                        result.file_existence_ratio,
                        [],
                        [],
                        llm_adjusted=True,
                    )

        return results

    # ------------------------------------------------------------------
    # Stage 1: File existence
    # ------------------------------------------------------------------

    def _stage_file_existence(
        self, proposal, existing_files_out: list
    ) -> float:
        """Check which affected files exist. Returns ratio of existing/total."""
        paths = list(proposal.affected_files) if proposal.affected_files else []

        # Fallback: gather file_path from implementation steps
        if not paths and proposal.implementation_steps:
            for step in proposal.implementation_steps:
                if step.file_path and step.file_path not in paths:
                    paths.append(step.file_path)

        if not paths:
            return 0.0

        existing = 0
        for p in paths:
            full = self.repo_path / p
            if full.exists():
                existing += 1
                existing_files_out.append(p)

        return existing / len(paths)

    # ------------------------------------------------------------------
    # Stage 2: Code content grep
    # ------------------------------------------------------------------

    def _stage_code_content(
        self, proposal, existing_files: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Extract identifiers from implementation steps, then check
        if they appear in the existing files. Returns (ratio, matches).
        """
        identifiers = self._extract_identifiers(proposal)
        if not identifiers or not existing_files:
            return 0.0, []

        matched = []
        for ident in identifiers:
            for fpath in existing_files:
                full = self.repo_path / fpath
                try:
                    content = full.read_text(errors="replace")
                    if ident in content:
                        matched.append(ident)
                        break
                except (OSError, IOError):
                    continue

        ratio = len(matched) / len(identifiers) if identifiers else 0.0
        return ratio, matched

    def _extract_identifiers(self, proposal) -> List[str]:
        """
        Parse class/function/constant names from proposal title
        and implementation step descriptions.
        """
        texts = [proposal.title]
        for step in proposal.implementation_steps:
            texts.append(step.description)

        identifiers = set()
        for text in texts:
            # Class names: "Create FooBar class" or "class FooBar"
            for m in re.finditer(r"[Cc]reate\s+(\w+)\s+class", text):
                identifiers.add(m.group(1))
            for m in re.finditer(r"class\s+(\w+)", text):
                identifiers.add(m.group(1))
            # Function names: "add foo_bar(" or "def foo_bar"
            for m in re.finditer(r"(?:add|create|implement)\s+(\w+)\s*\(", text):
                identifiers.add(m.group(1))
            for m in re.finditer(r"def\s+(\w+)", text):
                identifiers.add(m.group(1))
            # Config constants: 4+ char ALL_CAPS
            for m in re.finditer(r"\b([A-Z][A-Z_]{3,})\b", text):
                identifiers.add(m.group(1))

        # Filter noise
        noise = {"TODO", "NOTE", "FIXME", "HACK", "NONE", "TRUE", "FALSE", "NULL"}
        return [i for i in identifiers if i not in noise and len(i) >= 3]

    # ------------------------------------------------------------------
    # Stage 3: Git history
    # ------------------------------------------------------------------

    def _stage_git_history(self, proposal) -> Tuple[float, List[str]]:
        """
        Check recent git commits for file overlap + keyword overlap
        with the proposal. Returns (best_score, matching_subjects).
        """
        from config import app_config

        git_depth = getattr(app_config, "IMPL_TRACKING_GIT_DEPTH", 50)

        affected = set(proposal.affected_files) if proposal.affected_files else set()
        if not affected:
            return 0.0, []

        title_words = set(
            w.lower()
            for w in re.findall(r"\w+", proposal.title)
            if len(w) >= 4
        )

        try:
            cmd = [
                "git", "log",
                "--name-only",
                "--pretty=format:%H|||%s",
                f"-n{git_depth}",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.repo_path),
                timeout=10,
            )
            if result.returncode != 0:
                return 0.0, []

            output = result.stdout.strip()
            if not output:
                return 0.0, []

        except (subprocess.SubprocessError, OSError):
            return 0.0, []

        # Parse commits
        best_score = 0.0
        matching_subjects = []
        current_subject = ""
        current_files = set()

        for line in output.split("\n"):
            line = line.strip()
            if "|||" in line:
                # Score previous commit
                if current_subject and current_files:
                    score = self._score_commit(
                        affected, title_words, current_files, current_subject
                    )
                    if score > 0.1:
                        matching_subjects.append(current_subject)
                    best_score = max(best_score, score)

                parts = line.split("|||", 1)
                current_subject = parts[1] if len(parts) > 1 else ""
                current_files = set()
            elif line:
                current_files.add(line)

        # Score final commit
        if current_subject and current_files:
            score = self._score_commit(
                affected, title_words, current_files, current_subject
            )
            if score > 0.1:
                matching_subjects.append(current_subject)
            best_score = max(best_score, score)

        return best_score, matching_subjects[:5]

    def _score_commit(
        self,
        affected: set,
        title_words: set,
        commit_files: set,
        subject: str,
    ) -> float:
        """Score a single commit's relevance to the proposal."""
        # File overlap
        file_overlap = len(affected & commit_files) / len(affected) if affected else 0.0

        # Keyword overlap
        subject_words = set(
            w.lower() for w in re.findall(r"\w+", subject) if len(w) >= 4
        )
        keyword_overlap = (
            len(title_words & subject_words) / len(title_words)
            if title_words
            else 0.0
        )

        return file_overlap * 0.7 + keyword_overlap * 0.3

    # ------------------------------------------------------------------
    # Stage 4: LLM judgment (borderline only)
    # ------------------------------------------------------------------

    async def _stage_llm_judgment(
        self, borderline: List[Tuple]
    ) -> Dict[str, float]:
        """
        Batch LLM call for borderline proposals (confidence 0.30-0.84).
        Returns {proposal_id: adjusted_confidence}.
        """
        if not self.model_manager:
            return {}

        # Cap at 5 proposals per call
        batch = borderline[:5]
        lines = []
        for i, (proposal, result) in enumerate(batch):
            files_str = ", ".join(proposal.affected_files[:5]) if proposal.affected_files else "none"
            lines.append(
                f"{i+1}. \"{proposal.title}\" | files: {files_str} | "
                f"file_exists: {result.file_existence_ratio:.0%} | "
                f"code_match: {result.code_match_ratio:.0%} | "
                f"git_match: {result.git_match_score:.0%}"
            )

        prompt = (
            "For each proposal below, estimate the probability (0.0-1.0) that it has "
            "already been implemented in the codebase based on the evidence.\n\n"
            + "\n".join(lines)
            + "\n\nRespond with JSON only: {\"1\": 0.75, \"2\": 0.40, ...}"
        )

        try:
            response = await self.model_manager.generate_once(
                prompt,
                system_prompt="You are a code analysis assistant. Respond with JSON only.",
                max_tokens=200,
                temperature=0.0,
            )

            import json
            # Try to extract JSON from response
            text = response.strip()
            # Handle possible markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)

            adjustments = {}
            for i, (proposal, _result) in enumerate(batch):
                key = str(i + 1)
                if key in parsed:
                    val = float(parsed[key])
                    adjustments[proposal.id] = max(0.0, min(1.0, val))

            return adjustments

        except Exception as e:
            logger.debug(f"[ImplementationDetector] LLM judgment failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self, file_ratio: float, code_ratio: float, git_score: float
    ) -> float:
        """Weighted composite: files 0.35, code 0.35, git 0.30."""
        raw = file_ratio * 0.35 + code_ratio * 0.35 + git_score * 0.30
        return max(0.0, min(1.0, raw))

    def _confidence_to_status(self, confidence: float) -> str:
        """Map confidence to human-readable status."""
        from config import app_config

        threshold_confirmed = getattr(
            app_config, "IMPL_TRACKING_CONFIDENCE_CONFIRMED", 0.85
        )
        threshold_likely = getattr(
            app_config, "IMPL_TRACKING_CONFIDENCE_LIKELY", 0.60
        )

        if confidence >= threshold_confirmed:
            return "confirmed"
        elif confidence >= threshold_likely:
            return "likely"
        elif confidence >= 0.30:
            return "uncertain"
        else:
            return "not_implemented"

    def _build_evidence(
        self,
        file_ratio: float,
        code_matches: List[str],
        git_matches: List[str],
        llm_adjusted: bool,
    ) -> str:
        """Build human-readable evidence string, capped at 500 chars."""
        parts = []
        parts.append(f"Files: {file_ratio:.0%} exist")

        if code_matches:
            parts.append(f"Code: found {', '.join(code_matches[:5])}")

        if git_matches:
            subjects = "; ".join(git_matches[:3])
            parts.append(f"Git: {subjects}")

        if llm_adjusted:
            parts.append("(LLM-adjusted)")

        evidence = " | ".join(parts)
        return evidence[:500]

    def _should_skip_cooldown(self, proposal) -> bool:
        """True if proposal was checked within cooldown period."""
        from config import app_config

        cooldown = getattr(app_config, "IMPL_TRACKING_COOLDOWN", 86400)
        last = getattr(proposal, "last_tracked_at", None)
        if not last:
            return False
        return (time.time() - last) < cooldown
