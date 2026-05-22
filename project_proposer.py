"""
project_proposer.py - Standalone feature proposal generator for any project.

Module Contract
- Purpose: Scan a project's documentation, git history, and file structure,
  then use an LLM to generate structured feature proposals with reasoning,
  implementation steps, and priority rankings. Single-file, drop-in,
  zero required external dependencies.
- Inputs:
  - ProjectProposer(repo_path, llm, context_files, max_file_chars, max_proposals):
    constructor with configurable context sources and LLM callable
  - llm: Callable with signature (prompt, system_prompt, max_tokens, temperature) -> str
  - openai_llm(api_key, model, base_url): factory for OpenAI-compatible LLM callable
  - anthropic_llm(api_key, model): factory for Anthropic LLM callable
  - generate(extra_context, max_proposals): main generation entry point
- Outputs:
  - list[Proposal]: validated, priority-sorted proposal dataclasses
  - Each Proposal has: title, proposal_type, priority (1-10), reasoning,
    description, implementation_steps, affected_files, tags,
    estimated_complexity, requires_tests
  - JSON-serializable via Proposal.to_dict() / Proposal.from_dict()
  - CLI outputs formatted text or JSON to stdout
- Key behaviors:
  - Context gathering: reads configured doc files (README.md, CLAUDE.md, etc.),
    git log --oneline -n 15, top-level directory listing, project metadata
    (pyproject.toml, package.json, Cargo.toml, go.mod, pom.xml)
  - Graceful degradation: missing files, unavailable git, no metadata all skipped
  - LLM response parsing: handles JSON arrays, markdown code fences,
    and line-delimited JSON as fallback
  - Proposal validation: title min length, type/complexity enum clamping,
    priority range enforcement (1-10), tag cap (10)
  - Results sorted by priority (highest first)
- Side effects:
  - Subprocess call to git for recent commits (read-only, 5s timeout)
  - LLM API call via user-provided callable
  - File reads (project docs) — read-only
  - CLI prints to stdout/stderr
- Dependencies:
  - Required: Python 3.10+ standard library only
  - Optional: openai SDK (for openai_llm adapter), anthropic SDK (for anthropic_llm adapter)

Usage (Python API):
    from project_proposer import ProjectProposer, openai_llm

    proposer = ProjectProposer(llm=openai_llm())
    proposals = proposer.generate()

    for p in proposals:
        print(f"[{p.priority}] {p.title}")
        print(f"  {p.reasoning}")

Usage (CLI):
    # OpenAI (reads OPENAI_API_KEY from env)
    python project_proposer.py --provider openai --model gpt-4o

    # Anthropic (reads ANTHROPIC_API_KEY from env)
    python project_proposer.py --provider anthropic --model claude-sonnet-4-20250514

    # Custom repo path + extra context
    python project_proposer.py --repo /path/to/project --provider openai \\
        --extra-context "We're planning a v2 release focused on performance"

    # Output as JSON
    python project_proposer.py --provider openai --json

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

__version__ = "0.1.0"

logger = logging.getLogger("project_proposer")

# Type alias for the LLM callable.
# Signature: (prompt, system_prompt, max_tokens, temperature) -> response_text
LLMCallable = Callable[..., str]

# Default doc files to scan for project context (checked in order, all found are used).
DEFAULT_CONTEXT_FILES: list[str] = [
    "CLAUDE.md",
    "README.md",
    "CONTRIBUTING.md",
    "ARCHITECTURE.md",
    "docs/ARCHITECTURE.md",
    "GOALS.md",
    "docs/GOALS.md",
    "docs/DESIGN.md",
    "DESIGN.md",
]


# ############################################################################
#
#  Data models
#
# ############################################################################

@dataclass
class ImplementationStep:
    """A single step in a proposal's implementation plan."""
    order: int = 1
    description: str = ""
    file_path: str = ""
    action: str = "modify"  # create, modify, delete, test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationStep:
        return cls(
            order=int(data.get("order", 1)),
            description=data.get("description", ""),
            file_path=data.get("file_path", ""),
            action=data.get("action", "modify"),
        )


@dataclass
class Proposal:
    """A proposed feature or improvement for the project."""
    title: str = ""
    proposal_type: str = "feature"  # feature, refactor, bugfix, test, docs, infra
    priority: int = 5              # 1-10 (10 = highest)
    reasoning: str = ""
    description: str = ""
    implementation_steps: list[ImplementationStep] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high
    requires_tests: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["implementation_steps"] = [s.to_dict() for s in self.implementation_steps]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        steps_raw = data.get("implementation_steps", [])
        steps = [
            ImplementationStep.from_dict(s) if isinstance(s, dict) else s
            for s in steps_raw
        ]
        return cls(
            title=data.get("title", ""),
            proposal_type=data.get("proposal_type", "feature"),
            priority=max(1, min(10, int(data.get("priority", 5)))),
            reasoning=data.get("reasoning", ""),
            description=data.get("description", ""),
            implementation_steps=steps,
            affected_files=data.get("affected_files", []),
            tags=data.get("tags", [])[:10],
            estimated_complexity=data.get("estimated_complexity", "medium"),
            requires_tests=bool(data.get("requires_tests", True)),
        )

    def summary(self) -> str:
        """One-line summary for display."""
        kind = self.proposal_type.upper()
        return f"[{kind}] (P{self.priority}) {self.title}"


# ############################################################################
#
#  LLM adapters
#
# ############################################################################

def openai_llm(
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
) -> LLMCallable:
    """Create an LLM callable using the OpenAI SDK.

    Reads ``OPENAI_API_KEY`` from environment if *api_key* not provided.
    Set *base_url* to use OpenAI-compatible providers (e.g. Ollama, vLLM).
    """
    try:
        import openai  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install openai  (required for openai_llm adapter)")

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    def call(
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    return call


def anthropic_llm(
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> LLMCallable:
    """Create an LLM callable using the Anthropic SDK.

    Reads ``ANTHROPIC_API_KEY`` from environment if *api_key* not provided.
    """
    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install anthropic  (required for anthropic_llm adapter)")

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**kwargs)

    def call(
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            req["system"] = system_prompt
        response = client.messages.create(**req)
        return response.content[0].text if response.content else ""

    return call


# ############################################################################
#
#  Project proposer
#
# ############################################################################

class ProjectProposer:
    """Scans a project and generates feature proposals via an LLM.

    Args:
        repo_path: Path to the project root.  Defaults to current directory.
        llm: Callable with signature
            ``(prompt, system_prompt, max_tokens, temperature) -> str``.
            Use ``openai_llm()`` or ``anthropic_llm()`` for convenience.
        context_files: List of relative file paths to read for project context.
            Defaults to common doc files (README.md, CLAUDE.md, etc.).
        max_file_chars: Maximum characters to read per context file.
        max_proposals: Maximum number of proposals to return.
    """

    def __init__(
        self,
        repo_path: str | Path = ".",
        llm: LLMCallable | None = None,
        context_files: list[str] | None = None,
        max_file_chars: int = 8000,
        max_proposals: int = 5,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.llm = llm
        self.context_files = context_files or list(DEFAULT_CONTEXT_FILES)
        self.max_file_chars = max_file_chars
        self.max_proposals = max_proposals

    # ------------------------------------------------------------------
    # Context gathering
    # ------------------------------------------------------------------

    def gather_context(self) -> dict[str, str]:
        """Gather project context from docs, git history, and file structure.

        Returns a dict mapping source names to their text content.
        Gracefully skips anything that's missing or unreadable.
        """
        context: dict[str, str] = {}

        # 1. Read configured context files
        for rel_path in self.context_files:
            full_path = self.repo_path / rel_path
            if full_path.exists() and full_path.is_file():
                try:
                    text = full_path.read_text(encoding="utf-8")
                    if text.strip():
                        context[rel_path] = text[: self.max_file_chars]
                except Exception as e:
                    logger.debug(f"Could not read {rel_path}: {e}")

        # 2. Recent git commits
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-n", "15"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                context["git_log"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.debug("Git not available, skipping recent commits")

        # 3. Top-level project structure
        try:
            items = sorted(self.repo_path.iterdir())
            dirs = [p.name + "/" for p in items if p.is_dir() and not p.name.startswith(".")]
            files = [p.name for p in items if p.is_file() and not p.name.startswith(".")]
            if dirs or files:
                parts = []
                if dirs:
                    parts.append("Directories: " + ", ".join(dirs[:30]))
                if files:
                    parts.append("Files: " + ", ".join(files[:20]))
                context["project_structure"] = "\n".join(parts)
        except OSError:
            pass

        # 4. Project metadata (first match wins)
        for meta_file in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml"):
            meta_path = self.repo_path / meta_file
            if meta_path.exists():
                try:
                    text = meta_path.read_text(encoding="utf-8")
                    if text.strip():
                        context[meta_file] = text[:2000]
                except Exception:
                    pass
                break

        return context

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, context: dict[str, str], extra_context: str = "") -> str:
        """Assemble the LLM prompt from gathered context."""
        sections: list[str] = []

        # Group context by type for cleaner prompt structure
        doc_keys = [k for k in context if k not in ("git_log", "project_structure")]
        for key in doc_keys:
            label = key.replace("/", " / ").replace("_", " ").title()
            sections.append(f"## {label}\n{context[key]}")

        if "project_structure" in context:
            sections.append(f"## Project Structure\n{context['project_structure']}")

        if "git_log" in context:
            sections.append(f"## Recent Git Commits\n{context['git_log']}")

        if extra_context:
            sections.append(f"## Additional Context\n{extra_context}")

        context_block = "\n\n".join(sections)

        prompt = (
            "You are a senior software architect reviewing this project. "
            "Your job is to propose HIGH-IMPACT changes that push the project forward - "
            "not just fix what exists, but imagine what SHOULD exist.\n\n"
            "You have full creative authority to propose:\n"
            "- Entirely NEW features and capabilities\n"
            "- New external API integrations\n"
            "- New data models, storage patterns, or architectures\n"
            "- New modules or subsystems from scratch\n"
            "- Refactors, bug fixes, tests, docs, and infra improvements\n\n"
            "Each proposal should be a JSON object with these fields:\n"
            '{"title": "short title", '
            '"proposal_type": "feature|refactor|bugfix|test|docs|infra", '
            '"priority": 1-10, '
            '"reasoning": "why this change is needed", '
            '"description": "detailed description", '
            '"implementation_steps": [{"order": 1, "description": "step desc", '
            '"file_path": "path/to/file", "action": "create|modify|delete|test"}], '
            '"affected_files": ["path/to/file"], '
            '"tags": ["keyword1", "keyword2"], '
            '"estimated_complexity": "low|medium|high", '
            '"requires_tests": true}\n\n'
            "Rules:\n"
            f"- Generate 3-{self.max_proposals} diverse proposals\n"
            "- Prioritize proposals aligned with project goals (if provided)\n"
            "- Be specific: name concrete files, APIs, libraries, and patterns\n"
            "- For new features, use action 'create' and propose real file paths\n"
            "- Output ONLY a JSON array of proposal objects, no other text\n"
            "- If nothing meaningful can be proposed, output an empty array []\n\n"
            f"PROJECT CONTEXT:\n{context_block}\n\n"
            "Proposals (JSON array only):"
        )

        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> list[dict[str, Any]]:
        """Parse LLM response into a list of proposal dicts.

        Handles markdown code fences, JSON arrays, and line-delimited JSON.
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

        # Fall back to line-by-line JSON
        results: list[dict[str, Any]] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return results

    @staticmethod
    def _validate_proposal(data: dict[str, Any]) -> Proposal | None:
        """Parse a raw dict into a validated Proposal. Returns None if invalid."""
        try:
            title = data.get("title", "").strip()
            if not title or len(title) < 3:
                return None

            ptype = data.get("proposal_type", "feature").lower()
            if ptype not in ("feature", "refactor", "bugfix", "test", "docs", "infra"):
                ptype = "feature"

            complexity = data.get("estimated_complexity", "medium").lower()
            if complexity not in ("low", "medium", "high"):
                complexity = "medium"

            steps = []
            for step_data in data.get("implementation_steps", []):
                if isinstance(step_data, dict):
                    try:
                        steps.append(ImplementationStep.from_dict(step_data))
                    except Exception:
                        continue

            return Proposal(
                title=title,
                proposal_type=ptype,
                priority=max(1, min(10, int(data.get("priority", 5)))),
                reasoning=data.get("reasoning", ""),
                description=data.get("description", ""),
                implementation_steps=steps,
                affected_files=data.get("affected_files", []),
                tags=data.get("tags", [])[:10],
                estimated_complexity=complexity,
                requires_tests=bool(data.get("requires_tests", True)),
            )
        except Exception as e:
            logger.debug(f"Failed to parse proposal: {e}")
            return None

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate(
        self,
        extra_context: str = "",
        max_proposals: int | None = None,
    ) -> list[Proposal]:
        """Generate feature proposals for the project.

        Args:
            extra_context: Additional text to include in the LLM prompt
                (e.g. team priorities, constraints, upcoming deadlines).
            max_proposals: Override the default max proposals.

        Returns:
            List of validated Proposal objects, sorted by priority (highest first).

        Raises:
            ValueError: If no LLM callable was provided.
        """
        if not self.llm:
            raise ValueError(
                "No LLM callable provided. Pass llm= to the constructor, "
                "e.g. ProjectProposer(llm=openai_llm()) or "
                "ProjectProposer(llm=anthropic_llm())"
            )

        limit = max_proposals or self.max_proposals

        # Gather context
        context = self.gather_context()

        if not any(context.values()) and not extra_context:
            logger.info("No project context found. Cannot generate proposals.")
            return []

        # Build prompt and call LLM
        prompt = self._build_prompt(context, extra_context)

        try:
            response = self.llm(
                prompt,
                system_prompt="You are a senior software architect. Output only valid JSON.",
                max_tokens=4000,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []

        if not response:
            return []

        # Parse and validate
        raw_proposals = self._parse_response(response)

        proposals: list[Proposal] = []
        for raw in raw_proposals:
            proposal = self._validate_proposal(raw)
            if proposal:
                proposals.append(proposal)
                if len(proposals) >= limit:
                    break

        # Sort by priority (highest first)
        proposals.sort(key=lambda p: p.priority, reverse=True)

        logger.info(f"Generated {len(proposals)} proposals from {len(raw_proposals)} raw")
        return proposals


# ############################################################################
#
#  CLI
#
# ############################################################################

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI-powered feature proposals for your project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python project_proposer.py --provider openai\n"
            "  python project_proposer.py --provider anthropic --model claude-sonnet-4-20250514\n"
            "  python project_proposer.py --provider openai --model gpt-4o --json\n"
            "  python project_proposer.py --repo ~/myproject --provider openai "
            '--extra-context "Focus on API performance"\n'
        ),
    )
    parser.add_argument(
        "--repo", default=".", help="Path to the project root (default: current directory)"
    )
    parser.add_argument(
        "--provider", required=True, choices=["openai", "anthropic"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: gpt-4o for openai, claude-sonnet-4-20250514 for anthropic)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (default: read from OPENAI_API_KEY or ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--max-proposals", type=int, default=5, help="Maximum proposals to generate (default: 5)",
    )
    parser.add_argument(
        "--extra-context", default="",
        help="Additional context to include in the prompt",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output proposals as JSON instead of formatted text",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Create LLM adapter
    if args.provider == "openai":
        model = args.model or "gpt-4o"
        llm = openai_llm(api_key=args.api_key, model=model)
    else:
        model = args.model or "claude-sonnet-4-20250514"
        llm = anthropic_llm(api_key=args.api_key, model=model)

    # Generate
    proposer = ProjectProposer(
        repo_path=args.repo,
        llm=llm,
        max_proposals=args.max_proposals,
    )

    print(f"Scanning project at {proposer.repo_path} ...", file=sys.stderr)
    proposals = proposer.generate(extra_context=args.extra_context)

    if not proposals:
        print("No proposals generated.", file=sys.stderr)
        sys.exit(0)

    # Output
    if args.json_output:
        print(json.dumps([p.to_dict() for p in proposals], indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"  {len(proposals)} Proposal(s) for {proposer.repo_path.name}")
        print(f"{'=' * 60}\n")
        for i, p in enumerate(proposals, 1):
            print(f"  {i}. {p.summary()}")
            print(f"     Complexity: {p.estimated_complexity}")
            if p.reasoning:
                print(f"     Why: {p.reasoning[:120]}")
            if p.description:
                desc = p.description[:200]
                print(f"     What: {desc}")
            if p.affected_files:
                print(f"     Files: {', '.join(p.affected_files[:5])}")
            if p.implementation_steps:
                print(f"     Steps: {len(p.implementation_steps)}")
                for s in p.implementation_steps[:3]:
                    print(f"       {s.order}. [{s.action}] {s.description[:80]}")
                if len(p.implementation_steps) > 3:
                    print(f"       ... and {len(p.implementation_steps) - 3} more")
            if p.tags:
                print(f"     Tags: {', '.join(p.tags)}")
            print()


if __name__ == "__main__":
    _cli()
