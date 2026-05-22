# Project Proposer - Architecture & Skeleton

**Purpose**: Compressed architectural overview of `project_proposer.py` — a
standalone feature proposal generator that scans any project and uses an LLM
to produce structured, prioritized improvement proposals.

**Last Updated**: 2026-05-22

---

## 1. Architecture Overview

```
PROJECT ROOT
    |
    v
[Context Gathering]
    |
    +-- Doc files: README.md, CLAUDE.md, CONTRIBUTING.md, ARCHITECTURE.md, ...
    +-- Git history: git log --oneline -n 15
    +-- Project structure: top-level dirs + files listing
    +-- Project metadata: pyproject.toml / package.json / Cargo.toml / go.mod / pom.xml
    |
    v
[Prompt Assembly]
    |
    +-- Grouped context sections (docs, structure, git, metadata)
    +-- Extra user-provided context (optional)
    +-- Creative mandate prompt: "propose HIGH-IMPACT changes"
    +-- JSON output format specification
    |
    v
[LLM Call]                               <-- via user-provided callable
    |                                         (openai_llm, anthropic_llm, or custom)
    v
[Response Parsing]
    |
    +-- Try JSON array
    +-- Try markdown code fence extraction
    +-- Fallback: line-delimited JSON
    |
    v
[Validation & Sorting]
    |
    +-- Title min length (3 chars)
    +-- Type/complexity enum clamping
    +-- Priority range enforcement (1-10)
    +-- Tag cap (10)
    +-- Sort by priority descending
    |
    v
list[Proposal]                           <-- structured output
```

---

## 2. Module Structure

```
project_proposer.py (single file, ~670 lines)
|
+-- Constants
|   DEFAULT_CONTEXT_FILES: list[str]     # 9 common doc file paths
|
+-- Data Models
|   @dataclass ImplementationStep
|   |   order: int
|   |   description: str
|   |   file_path: str
|   |   action: str                      # create, modify, delete, test
|   |   to_dict() -> dict
|   |   from_dict(data) -> ImplementationStep
|   |
|   @dataclass Proposal
|       title: str
|       proposal_type: str               # feature, refactor, bugfix, test, docs, infra
|       priority: int                    # 1-10
|       reasoning: str
|       description: str
|       implementation_steps: list[ImplementationStep]
|       affected_files: list[str]
|       tags: list[str]
|       estimated_complexity: str        # low, medium, high
|       requires_tests: bool
|       to_dict() -> dict
|       from_dict(data) -> Proposal
|       summary() -> str                 # "[FEATURE] (P8) Title"
|
+-- LLM Adapters
|   openai_llm(api_key, model, base_url) -> LLMCallable
|   anthropic_llm(api_key, model) -> LLMCallable
|
+-- ProjectProposer
|   __init__(repo_path, llm, context_files, max_file_chars, max_proposals)
|   gather_context() -> dict[str, str]
|   _build_prompt(context, extra_context) -> str
|   _parse_response(response) -> list[dict]      # static
|   _validate_proposal(data) -> Proposal | None   # static
|   generate(extra_context, max_proposals) -> list[Proposal]
|
+-- CLI
    _cli() -> None                       # argparse entry point
    __main__ block
```

---

## 3. Context Gathering Pipeline

`gather_context()` returns a `dict[str, str]` mapping source names to text.
All sources are optional — missing files and unavailable git are silently skipped.

```
Step  Source                     Key in dict           Cap        Notes
----  ------                     -----------           ---        -----
 1    Configured doc files       rel_path (e.g.        8000 chars  DEFAULT_CONTEXT_FILES or
      (README.md, CLAUDE.md,     "README.md")                     user-provided list
      CONTRIBUTING.md, ...)
 2    Git log                    "git_log"             15 commits  subprocess, 5s timeout
 3    Directory listing          "project_structure"   30 dirs     top-level only,
                                                      + 20 files  hides dotfiles
 4    Project metadata           meta filename         2000 chars  first match wins:
                                 (e.g. "pyproject.     pyproject.toml, package.json,
                                 toml")                Cargo.toml, go.mod, pom.xml
```

### Customizing context sources

```python
# Override the doc file list
proposer = ProjectProposer(
    context_files=["docs/ROADMAP.md", "docs/API.md", "CHANGELOG.md"],
    max_file_chars=12000,
)

# Or add extra context at generation time
proposals = proposer.generate(
    extra_context="We're migrating from REST to GraphQL next quarter"
)
```

---

## 4. LLM Interface

The proposer accepts any callable matching this signature:

```python
def llm(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 4000,
    temperature: float = 0.7,
) -> str:
    ...
```

### Built-in adapters

| Adapter          | SDK Required | Default Model               | Auth                    |
|------------------|--------------|-----------------------------|-------------------------|
| `openai_llm()`   | `openai`     | `gpt-4o`                    | `OPENAI_API_KEY` env    |
| `anthropic_llm()` | `anthropic` | `claude-sonnet-4-20250514`  | `ANTHROPIC_API_KEY` env |

Both adapters support explicit `api_key=` override. `openai_llm()` also accepts
`base_url=` for OpenAI-compatible providers (Ollama, vLLM, Together, etc.).

### Custom adapter example

```python
import requests

def my_llm(prompt, system_prompt="", max_tokens=4000, temperature=0.7):
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    })
    return resp.json()["response"]

proposer = ProjectProposer(llm=my_llm)
```

---

## 5. Response Parsing Strategy

`_parse_response()` handles three LLM output formats, tried in order:

```
1. JSON array:        [{"title": "...", ...}, {"title": "...", ...}]
2. Code-fenced JSON:  ```json\n[...]\n```
3. Line-delimited:    {"title": "..."}\n{"title": "..."}
```

This resilience is necessary because different LLMs (and different temperatures)
produce varying output formats even with explicit "output JSON only" instructions.

---

## 6. Proposal Validation Rules

`_validate_proposal()` applies these checks:

| Field                  | Rule                                         | On failure        |
|------------------------|----------------------------------------------|-------------------|
| `title`                | Must be string, stripped, length >= 3         | Return None       |
| `proposal_type`        | Must be one of 6 enum values                 | Default "feature" |
| `priority`             | Clamped to [1, 10]                           | Clamped           |
| `estimated_complexity`  | Must be low/medium/high                     | Default "medium"  |
| `tags`                 | Max 10 items                                 | Truncated         |
| `implementation_steps` | Each must be a valid dict                    | Invalid skipped   |
| `requires_tests`       | Coerced to bool                              | Default True      |

---

## 7. Data Flow: generate()

```python
def generate(extra_context="", max_proposals=None):
    # 1. Validate LLM callable exists
    if not self.llm: raise ValueError(...)

    # 2. Gather project context (all sources, graceful skip)
    context = self.gather_context()

    # 3. Build prompt (context sections + format spec + creative mandate)
    prompt = self._build_prompt(context, extra_context)

    # 4. Call LLM
    response = self.llm(prompt, system_prompt=..., max_tokens=4000, temperature=0.7)

    # 5. Parse response (JSON array / fenced / line-delimited)
    raw_proposals = self._parse_response(response)

    # 6. Validate each, collect up to limit
    proposals = [self._validate_proposal(r) for r in raw_proposals if ...]

    # 7. Sort by priority (highest first)
    proposals.sort(key=lambda p: p.priority, reverse=True)

    return proposals
```

---

## 8. CLI Interface

```
python project_proposer.py [OPTIONS]

Required:
  --provider {openai,anthropic}    LLM provider

Optional:
  --repo PATH                      Project root (default: .)
  --model NAME                     Model name (default: provider-specific)
  --api-key KEY                    API key (default: from env var)
  --max-proposals N                Max proposals (default: 5)
  --extra-context TEXT             Additional prompt context
  --json                           Output as JSON instead of formatted text
  --verbose / -v                   Enable debug logging
```

### Output formats

**Text (default):**
```
============================================================
  3 Proposal(s) for myproject
============================================================

  1. [FEATURE] (P9) Add WebSocket support for real-time updates
     Complexity: medium
     Why: Current polling causes unnecessary latency...
     Files: src/ws.py, src/handlers.py
     Steps: 3
       1. [create] Create WebSocket handler module
       2. [modify] Add WS route to main app
       3. [test] Add integration tests
```

**JSON (`--json`):**
```json
[
  {
    "title": "Add WebSocket support for real-time updates",
    "proposal_type": "feature",
    "priority": 9,
    ...
  }
]
```

---

## 9. Key Design Decisions

### Why stdlib dataclasses instead of Pydantic?

Zero-dependency grab-and-go. Pydantic would add validation power but also a
hard dependency. The validation in `_validate_proposal()` is simple enough
to handle manually, and `dataclasses.asdict()` gives us JSON serialization.

### Why sync instead of async?

Most grab-and-go users won't be in an async context. The only I/O is one
LLM call and file reads — neither benefits materially from async in a
single-file tool. Users in async contexts can wrap with `asyncio.to_thread()`.

### Why callable instead of client object?

A callable `(prompt, system_prompt, max_tokens, temperature) -> str` is the
simplest possible LLM interface. It works with any provider, any SDK version,
any wrapper. The factory functions (`openai_llm`, `anthropic_llm`) are
convenience — the core never imports either SDK.

### Why scan for multiple doc formats?

Projects are diverse. A Go project has `go.mod`, a Python project has
`pyproject.toml`, a Node project has `package.json`. The proposer should
work out of the box for all of them without configuration.

---

## 10. Invariants

1. **No writes**: The proposer never writes to the filesystem. All operations are
   read-only (doc files, git log, directory listing).
2. **No network without LLM**: The only network call is the LLM callable provided
   by the user. Context gathering is fully local.
3. **Graceful degradation**: An empty project (no docs, no git, no metadata)
   returns an empty proposal list with a log message — never raises.
4. **Deterministic parsing**: `_parse_response()` always returns a list (possibly
   empty). Invalid JSON entries are silently skipped, never raise.
5. **Provider-agnostic**: The core `ProjectProposer` class never imports any
   LLM SDK. Provider coupling is isolated to the factory functions.
