# Standalone Utilities

Grab-and-go single-file utilities extracted from the [Daemon](https://github.com/your-repo/daemon) project. Each file is fully self-contained — drop it into your project and use it directly.

**Architecture docs**: See `docs/AGENT_GUARD.md` and `docs/PROJECT_PROPOSER.md` for full architecture overviews, data flow diagrams, design decisions, and invariants.

**Tests**: `pytest standalone/tests/` — 213 tests, stdlib-only, no API keys needed.

## agent_guard.py

**Agent safety guard for Python projects.** Protects critical paths from AI agent operations by monkey-patching destructive filesystem calls (`os.remove`, `shutil.rmtree`, etc.) and classifying git/shell commands.

**Dependencies:** None (Python 3.10+ stdlib only)

### Quick start

```python
from agent_guard import activate, agent_mode

# Activate at startup
activate(
    repo_root=".",
    protected_dirs={".git", "src", "config", "data"},
    protected_files={".env", "main.py"},
)

# Wrap agent operations
with agent_mode():
    # Protected paths raise PermissionError:
    os.remove("config/settings.yaml")   # BLOCKED
    shutil.rmtree(".git/")              # BLOCKED
    os.remove("/tmp/scratch.txt")       # ALLOWED (outside repo)

# Outside agent_mode(), everything passes through normally
```

### Command classification (no monkey-patching needed)

```python
from agent_guard import classify_git_command, classify_shell_command

classify_git_command(["push"])
# {"subcmd": "push", "destructive": True, "reason": "git push is always destructive"}

classify_shell_command(["rm", "-rf", "src/"], repo_root=".", protected_dirs={"src"})
# {"command": "rm", "destructive": True, "severity": "protected", ...}
```

### Unlock mechanism

Protected-path blocks can be overridden when explicitly needed:
- Set `ALLOW_DESTRUCTIVE_OPS=1` in environment
- Create `.agent_allow_destructive_once` in repo root (name is configurable)

Always-blocked targets (`.`, `..`, `/`, `~`, `*`) can **never** be unlocked.

---

## project_proposer.py

**AI-powered feature proposal generator.** Scans your project's docs, git history, and structure, then uses an LLM to propose concrete features, refactors, and improvements.

**Dependencies:** None required (Python 3.10+ stdlib). Optional: `openai` or `anthropic` SDK.

### Quick start (Python)

```python
from project_proposer import ProjectProposer, openai_llm

proposer = ProjectProposer(llm=openai_llm())  # reads OPENAI_API_KEY from env
proposals = proposer.generate()

for p in proposals:
    print(p.summary())
    # [FEATURE] (P8) Add WebSocket support for real-time updates
```

### Quick start (CLI)

```bash
# OpenAI
python project_proposer.py --provider openai --model gpt-4o

# Anthropic
python project_proposer.py --provider anthropic --model claude-sonnet-4-20250514

# JSON output
python project_proposer.py --provider openai --json

# With extra context
python project_proposer.py --provider openai \
    --extra-context "We're planning a v2 release focused on performance"
```

### Custom LLM backend

```python
def my_llm(prompt, system_prompt="", max_tokens=4000, temperature=0.7):
    # Call any LLM API you want
    return response_text

proposer = ProjectProposer(llm=my_llm)
```

### What it scans

By default, the proposer reads (if present):
- `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`
- `ARCHITECTURE.md`, `GOALS.md`, `DESIGN.md` (also in `docs/`)
- `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `pom.xml`
- `git log --oneline -n 15`
- Top-level directory listing

Override with `context_files=["my_doc.md", "docs/plan.md"]`.
