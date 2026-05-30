# knowledge/proposal_generator.py
"""
Goal-directed code proposal generator.

Module Contract
- Purpose: Generates CodeProposal objects by analyzing project goals, skeleton,
  recent git activity, and conversation context. Uses LLM to identify
  actionable improvements aligned with project direction. Also generates
  full implementation code for proposals to a staging directory.
- Inputs:
  - model_manager: ModelManager instance for LLM calls
  - repo_path: Path to the project repository
  - Optional context overrides (goals text, skeleton text)
  - CodeProposal objects for code generation
- Outputs:
  - List[CodeProposal]: Validated proposal objects ready for storage
  - Dict with generated files, staging directory path, and errors (code generation)
- Key behaviors:
  - Gathers project context (skeleton, goals, CLAUDE.md, QUICK_REFERENCE.md, recent commits)
  - Prompts LLM for structured JSON proposals (creative mandate for new features)
  - Robust parsing of LLM output (handles fences, arrays, line-delimited)
  - Graceful degradation when files/git unavailable
  - Full code generation: reads source files for modify actions, writes to staging dir
  - Creates _manifest.json for each generated proposal
- Side effects:
  - Subprocess call to git for recent commits (read-only)
  - LLM API call via model_manager
  - File writes to data/proposal_code/<proposal_id>/ staging directory (code generation only)
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from memory.code_proposal import (
    CodeProposal,
    ImplementationStep,
    ProposalSource,
    ProposalType,
)
from utils.logging_utils import get_logger

logger = get_logger("proposal_generator")


class GoalDirectedGenerator:
    """
    Generates code proposals by analyzing project context and goals.

    Uses ModelManager.generate_once() to produce structured JSON proposals,
    then parses and validates them into CodeProposal objects.
    """

    def __init__(
        self,
        model_manager=None,
        repo_path: str = ".",
        model_alias: str = "claude-opus-4.8",
        max_proposals: int = 5,
    ):
        self.model_manager = model_manager
        self.repo_path = Path(repo_path)
        self.model_alias = model_alias
        self.max_proposals = max_proposals

    def gather_context(self) -> Dict[str, str]:
        """
        Gather project context for proposal generation.

        Reads project skeleton, goals, CLAUDE.md, QUICK_REFERENCE.md,
        and recent git history. Handles missing files and unavailable
        git gracefully.

        Returns:
            Dict with keys: skeleton, goals, recent_commits, claude_md, quick_reference
        """
        context: Dict[str, str] = {
            "skeleton": "",
            "goals": "",
            "recent_commits": "",
            "claude_md": "",
            "quick_reference": "",
        }

        # Read project skeleton (skip "Core Components" section which is
        # ~112K chars of per-method docs redundant with CLAUDE.md + QUICK_REFERENCE.md)
        skeleton_paths = [
            self.repo_path / "docs" / "PROJECT_SKELETON.md",
            self.repo_path / "PROJECT_SKELETON.md",
        ]
        for path in skeleton_paths:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    context["skeleton"] = self._filter_skeleton_sections(text)
                    break
                except Exception as e:
                    logger.debug(f"Could not read skeleton at {path}: {e}")

        # Read goals file
        goals_paths = [
            self.repo_path / "docs" / "GOALS.md",
            self.repo_path / "GOALS.md",
        ]
        for path in goals_paths:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    context["goals"] = text[:4000]
                    break
                except Exception as e:
                    logger.debug(f"Could not read goals at {path}: {e}")

        # Read CLAUDE.md (project conventions and architecture)
        claude_md_paths = [
            self.repo_path / "CLAUDE.md",
            self.repo_path / "docs" / "CLAUDE.md",
        ]
        for path in claude_md_paths:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    context["claude_md"] = text
                    logger.debug(f"Loaded CLAUDE.md ({len(text)} chars)")
                    break
                except Exception as e:
                    logger.debug(f"Could not read CLAUDE.md at {path}: {e}")

        # Read QUICK_REFERENCE.md (API signatures and patterns)
        quick_ref_paths = [
            self.repo_path / "docs" / "QUICK_REFERENCE.md",
            self.repo_path / "QUICK_REFERENCE.md",
        ]
        for path in quick_ref_paths:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    context["quick_reference"] = text
                    logger.debug(f"Loaded QUICK_REFERENCE.md ({len(text)} chars)")
                    break
                except Exception as e:
                    logger.debug(f"Could not read QUICK_REFERENCE.md at {path}: {e}")

        # Recent git commits
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-n", "10"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                context["recent_commits"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.debug("Git not available, skipping recent commits")

        return context

    @staticmethod
    def _filter_skeleton_sections(text: str) -> str:
        """
        Filter PROJECT_SKELETON.md to exclude the massive 'Core Components'
        section (~112K chars of per-method docs) which is redundant with
        CLAUDE.md and QUICK_REFERENCE.md.

        Keeps: architecture overview, config, data flow, algorithms, file
        organization, patterns, testing, extension points, invariants, etc.
        """
        lines = text.split("\n")
        result_lines: List[str] = []
        skip = False

        for line in lines:
            # Detect top-level numbered sections: "## 2. Core Components"
            if line.startswith("## ") and ". " in line[:20]:
                # Check if this is the Core Components section
                if "Core Components" in line:
                    skip = True
                    continue
                else:
                    skip = False

            if not skip:
                result_lines.append(line)

        filtered = "\n".join(result_lines)
        logger.debug(
            f"Skeleton filtered: {len(text)} → {len(filtered)} chars "
            f"(removed Core Components section)"
        )
        return filtered

    def _build_prompt(self, context: Dict[str, str], extra_context: str = "") -> str:
        """Build the LLM prompt from gathered context."""
        sections = []

        if context.get("claude_md"):
            sections.append(f"## Project Conventions (CLAUDE.md)\n{context['claude_md']}")

        if context.get("quick_reference"):
            sections.append(f"## API Quick Reference\n{context['quick_reference']}")

        if context.get("skeleton"):
            sections.append(f"## Project Architecture\n{context['skeleton']}")

        if context.get("goals"):
            sections.append(f"## Project Goals\n{context['goals']}")

        if context.get("recent_commits"):
            sections.append(f"## Recent Git Commits\n{context['recent_commits']}")

        if extra_context:
            sections.append(f"## Additional Context\n{extra_context}")

        context_block = "\n\n".join(sections)

        prompt = (
            "You are a senior software architect and product visionary for this project. "
            "Your job is to propose HIGH-IMPACT changes that push the project forward — "
            "not just fix what exists, but imagine what SHOULD exist.\n\n"
            "You have full creative authority to propose:\n"
            "- Entirely NEW features and capabilities that don't exist yet\n"
            "- New external API integrations (any public API that adds value)\n"
            "- New data models, collections, or storage patterns\n"
            "- New modules, classes, or subsystems from scratch\n"
            "- Architectural changes that unlock future possibilities\n"
            "- Refactors, bug fixes, tests, docs, and infra improvements\n\n"
            "Think beyond the current codebase. If the project goals suggest a capability "
            "that requires building something entirely new, propose it with concrete "
            "implementation steps.\n\n"
            "Each proposal should be a JSON object on its own line with these fields:\n"
            '{"title": "short title", '
            '"proposal_type": "feature|refactor|bugfix|test|docs|infra", '
            '"priority": 1-10, '
            '"reasoning": "why this change is needed", '
            '"description": "detailed description", '
            '"implementation_steps": [{"order": 1, "description": "step desc", '
            '"file_path": "path/to/file.py", "action": "create|modify|delete|test"}], '
            '"affected_files": ["path/to/file.py"], '
            '"tags": ["keyword1", "keyword2"], '
            '"estimated_complexity": "low|medium|high", '
            '"requires_tests": true}\n\n'
            "Rules:\n"
            "- Generate 3-5 diverse proposals per call\n"
            "- Prioritize proposals aligned with project goals (if provided)\n"
            "- Propose NEW capabilities, not just improvements to existing ones\n"
            "- CRITICAL: If 'Existing Proposals' are listed below, do NOT propose anything "
            "similar. Propose completely different ideas in different areas of the codebase.\n"
            "- Be specific: name concrete files, APIs, libraries, and patterns\n"
            "- For new features, use action 'create' and propose real file paths\n"
            "- Include concrete file paths where possible\n"
            "- Output ONLY valid JSON lines, no other text\n"
            "- If nothing meaningful can be proposed, output nothing\n\n"
            f"PROJECT CONTEXT:\n{context_block}\n\n"
            "Proposals (JSON lines only):"
        )

        return prompt

    def _parse_response(self, response: str) -> List[dict]:
        """
        Parse LLM response into list of proposal dicts.
        Handles code fences, JSON arrays, and line-delimited JSON.
        """
        text = response.strip()
        if not text:
            return []

        # Strip markdown code fences
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try as JSON array first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Fall back to line-by-line
        results = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return results

    def _parse_proposal(self, data: dict) -> Optional[CodeProposal]:
        """
        Parse a raw dict into a validated CodeProposal.
        Returns None if the data is invalid.
        """
        try:
            title = data.get("title", "").strip()
            if not title or len(title) < 3:
                return None

            # Parse proposal type
            type_str = data.get("proposal_type", "feature").lower()
            try:
                proposal_type = ProposalType(type_str)
            except ValueError:
                proposal_type = ProposalType.FEATURE

            # Parse implementation steps
            steps = []
            for step_data in data.get("implementation_steps", []):
                if isinstance(step_data, dict):
                    try:
                        steps.append(ImplementationStep.from_dict(step_data))
                    except Exception:
                        continue

            # Parse complexity
            complexity = data.get("estimated_complexity", "medium").lower()
            if complexity not in ("low", "medium", "high"):
                complexity = "medium"

            proposal = CodeProposal(
                title=title,
                proposal_type=proposal_type,
                source=ProposalSource.GOAL_DIRECTED,
                priority=max(1, min(10, int(data.get("priority", 5)))),
                reasoning=data.get("reasoning", ""),
                description=data.get("description", ""),
                implementation_steps=steps,
                affected_files=data.get("affected_files", []),
                tags=data.get("tags", [])[:10],
                estimated_complexity=complexity,
                requires_tests=bool(data.get("requires_tests", True)),
            )

            return proposal

        except Exception as e:
            logger.debug(f"Failed to parse proposal: {e}")
            return None

    async def generate_proposals(
        self,
        extra_context: str = "",
        max_proposals: Optional[int] = None,
    ) -> List[CodeProposal]:
        """
        Generate code proposals from project context.

        Args:
            extra_context: Additional context to include in the prompt
            max_proposals: Override max proposals (defaults to self.max_proposals)

        Returns:
            List of validated CodeProposal objects
        """
        if not self.model_manager:
            logger.warning("[ProposalGenerator] No model_manager available")
            return []

        limit = max_proposals or self.max_proposals

        # Gather context
        context = self.gather_context()

        if not any(context.values()) and not extra_context:
            logger.info("[ProposalGenerator] No project context available")
            return []

        # Build and send prompt
        prompt = self._build_prompt(context, extra_context)

        try:
            response = await self.model_manager.generate_once(
                prompt,
                model_name=self.model_alias,
                system_prompt="You are a senior software architect. Output only valid JSON.",
                max_tokens=4000,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"[ProposalGenerator] LLM call failed: {e}")
            return []

        if not response:
            return []

        logger.debug(f"Raw proposal generation response:\n{response[:2000]}")

        # Parse response
        raw_proposals = self._parse_response(response)

        # Validate and convert
        proposals = []
        for raw in raw_proposals:
            proposal = self._parse_proposal(raw)
            if proposal:
                proposals.append(proposal)
                if len(proposals) >= limit:
                    break

        logger.info(f"[ProposalGenerator] Generated {len(proposals)} proposals from {len(raw_proposals)} raw")
        return proposals

    async def generate_proposals_with_context(
        self,
        pipeline_context: str,
        extra_context: str = "",
        max_proposals: Optional[int] = None,
    ) -> List[CodeProposal]:
        """
        Generate proposals using pre-gathered pipeline context.

        Unlike generate_proposals() which only has cold file reads + truncated
        conversation excerpts, this method accepts rich context pre-gathered
        from the Daemon retrieval pipeline (semantic memories, summaries,
        reflections, skills, user profile, etc.).

        The project skeleton and goals are still read from files for
        architectural grounding.

        Args:
            pipeline_context: Rich context from the Daemon retrieval pipeline
            extra_context: Additional context (e.g., dedup info)
            max_proposals: Override max proposals

        Returns:
            List of validated CodeProposal objects
        """
        if not self.model_manager:
            logger.warning("[ProposalGenerator] No model_manager available")
            return []

        limit = max_proposals or self.max_proposals

        # Still gather skeleton + goals + git log (architectural grounding)
        context = self.gather_context()

        # Merge pipeline context with any extra context
        combined_extra = pipeline_context
        if extra_context:
            combined_extra += f"\n\n{extra_context}"

        if not any(context.values()) and not combined_extra:
            logger.info("[ProposalGenerator] No project context available")
            return []

        # Build and send prompt (reuses existing _build_prompt)
        prompt = self._build_prompt(context, combined_extra)

        try:
            response = await self.model_manager.generate_once(
                prompt,
                model_name=self.model_alias,
                system_prompt="You are a senior software architect. Output only valid JSON.",
                max_tokens=4000,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"[ProposalGenerator] LLM call failed: {e}")
            return []

        if not response:
            return []

        logger.debug(f"Raw proposal generation response:\n{response[:2000]}")

        raw_proposals = self._parse_response(response)

        proposals = []
        for raw in raw_proposals:
            proposal = self._parse_proposal(raw)
            if proposal:
                proposals.append(proposal)
                if len(proposals) >= limit:
                    break

        logger.info(
            f"[ProposalGenerator] Generated {len(proposals)} proposals "
            f"from {len(raw_proposals)} raw (pipeline-enriched)"
        )
        return proposals

    # ------------------------------------------------------------------
    # Full code generation for a proposal
    # ------------------------------------------------------------------

    async def generate_code_for_proposal(
        self,
        proposal: CodeProposal,
        output_dir: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Generate actual implementation code for every step in a proposal.

        Reads current source files for 'modify' actions, generates complete
        file contents, and writes everything to a staging directory.

        Args:
            proposal: The CodeProposal to implement
            output_dir: Override output directory (default: data/proposal_code/<id>/)

        Returns:
            Dict with keys: proposal_id, output_dir, files (dict of path->content),
            errors (list of error strings)
        """
        if not self.model_manager:
            return {"proposal_id": proposal.id, "output_dir": "", "files": {}, "errors": ["No model_manager"]}

        staging = Path(output_dir) if output_dir else Path("data/proposal_code") / proposal.id
        staging.mkdir(parents=True, exist_ok=True)

        generated_files: Dict[str, str] = {}
        errors: List[str] = []

        for step in proposal.implementation_steps:
            file_path = step.file_path
            if not file_path:
                continue

            try:
                if step.action == "delete":
                    generated_files[file_path] = "# FILE MARKED FOR DELETION"
                    continue

                code = await self._generate_step_code(proposal, step)
                if code:
                    generated_files[file_path] = code
                else:
                    errors.append(f"Empty response for {file_path}")
            except Exception as e:
                errors.append(f"{file_path}: {e}")
                logger.warning(f"[CodeGen] Failed for {file_path}: {e}")

        # Write files to staging directory
        for rel_path, content in generated_files.items():
            out_path = staging / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")

        # Write manifest
        manifest = {
            "proposal_id": proposal.id,
            "title": proposal.title,
            "proposal_type": proposal.proposal_type.value,
            "description": proposal.description,
            "files_generated": list(generated_files.keys()),
            "errors": errors,
        }
        (staging / "_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        logger.info(
            f"[CodeGen] Generated {len(generated_files)} file(s) for '{proposal.title}' "
            f"-> {staging} ({len(errors)} errors)"
        )

        return {
            "proposal_id": proposal.id,
            "output_dir": str(staging),
            "files": generated_files,
            "errors": errors,
        }

    async def _generate_step_code(
        self, proposal: CodeProposal, step: "ImplementationStep"
    ) -> Optional[str]:
        """Generate code for a single implementation step."""
        file_path = step.file_path
        action = step.action

        # Read current file for modifications
        current_content = ""
        if action == "modify":
            try:
                source = self.repo_path / file_path
                if source.exists():
                    text = source.read_text(encoding="utf-8")
                    # Cap large files but keep enough context
                    current_content = text[:15000]
                    if len(text) > 15000:
                        current_content += f"\n\n# ... (truncated, full file is {len(text)} chars)"
            except Exception as e:
                logger.debug(f"[CodeGen] Could not read {file_path}: {e}")

        # Build prompt based on action type
        proposal_context = (
            f"PROPOSAL: {proposal.title}\n"
            f"GOAL: {proposal.description}\n"
            f"REASONING: {proposal.reasoning}"
        )

        if action == "modify" and current_content:
            prompt = (
                f"You are implementing a specific code change.\n\n"
                f"{proposal_context}\n\n"
                f"CURRENT FILE ({file_path}):\n"
                f"```\n{current_content}\n```\n\n"
                f"CHANGE REQUIRED:\n{step.description}\n\n"
                f"Output the COMPLETE modified file. Include ALL original code with "
                f"your changes applied. Do not use placeholders like '# rest of file unchanged' "
                f"— output the entire file content so it can be saved directly."
            )
        elif action == "create":
            prompt = (
                f"You are creating a new file as part of a code proposal.\n\n"
                f"{proposal_context}\n\n"
                f"FILE TO CREATE: {file_path}\n"
                f"PURPOSE: {step.description}\n\n"
                f"Output the COMPLETE file content. Follow Python best practices:\n"
                f"- Module docstring explaining purpose\n"
                f"- Proper imports\n"
                f"- Type hints\n"
                f"- Logging via `from utils.logging_utils import get_logger`\n"
                f"Output only the file content, no markdown fences."
            )
        elif action == "test":
            # Try to read the file being tested for context
            test_target = ""
            for af in proposal.affected_files:
                if af != file_path and not af.startswith("tests/"):
                    try:
                        target_path = self.repo_path / af
                        if target_path.exists():
                            text = target_path.read_text(encoding="utf-8")
                            test_target += f"\n\nSOURCE FILE ({af}):\n```\n{text[:8000]}\n```"
                    except Exception:
                        pass

            prompt = (
                f"You are writing tests for a code proposal.\n\n"
                f"{proposal_context}\n\n"
                f"TEST FILE TO CREATE: {file_path}\n"
                f"WHAT TO TEST: {step.description}\n"
                f"{test_target}\n\n"
                f"Output a COMPLETE pytest test file. Use:\n"
                f"- pytest + pytest-asyncio for async tests\n"
                f"- unittest.mock for mocking\n"
                f"- Descriptive test names\n"
                f"Output only the file content, no markdown fences."
            )
        else:
            # Fallback for unknown actions
            prompt = (
                f"You are generating code for a proposal step.\n\n"
                f"{proposal_context}\n\n"
                f"FILE: {file_path}\n"
                f"ACTION: {action}\n"
                f"DESCRIPTION: {step.description}\n\n"
                f"Output the complete file content."
            )

        response = await self.model_manager.generate_once(
            prompt,
            model_name=self.model_alias,
            system_prompt="You are an expert Python developer. Output only code, no explanations or markdown fences.",
            max_tokens=8000,
            temperature=0.2,
        )

        if not response:
            return None

        # Strip markdown fences if the LLM added them anyway
        code = response.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```python or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code
