# Agent Session Safety Layer

Defense-in-depth safeguards for AI coding agent sessions. Written instructions are not sufficient to prevent accidental data loss. This layer provides mechanical enforcement: pre-session snapshots, post-session audits, and a safe git wrapper.

## Why This Exists

On 2026-05-18, an AI coding agent ran `git restore` and destroyed uncommitted working-tree changes despite project instructions forbidding destructive git operations. This safety layer ensures:

1. Repo state is captured before agent work begins.
2. Destructive commands are blocked unless explicitly unlocked.
3. Post-session audits reveal what changed.
4. Accidental loss is recoverable from saved snapshots.

## Required Workflow

### Before any agent session

```bash
bash scripts/agent_session_start.sh
```

Creates a timestamped snapshot in `.agent_snapshots/YYYYMMDD_HHMMSS/` containing:
- `branch.txt` / `head.txt` — git ref state
- `status.txt` — working tree status
- `diff.patch` — unstaged changes
- `cached.diff` — staged changes
- `untracked.txt` — list of untracked files
- `reflog.txt` — recent commit history
- `manifest.json` — full filesystem manifest (path, size, mtime, sha256)
- `untracked_files.tar.gz` — archived untracked files (filtered, size-capped at 25MB per file)
- `skipped_untracked.txt` — files too large to archive

Snapshots rotate automatically (latest 10 kept).

### During agent work

Use the safe git wrapper for any git operations:

```bash
bash scripts/safe_git.sh status
bash scripts/safe_git.sh diff
bash scripts/safe_git.sh log --oneline -5
bash scripts/safe_git.sh add file.py
bash scripts/safe_git.sh commit -m "message"
```

### After agent work

```bash
bash scripts/agent_session_audit.sh
```

Reports: branch/HEAD changes, modified/deleted/staged/untracked files, manifest diff (added/deleted/modified files), suspicious large files.

## Blocked Commands

The safe git wrapper (`scripts/safe_git.sh`) blocks these by default:

| Command | Reason |
|---------|--------|
| `git restore` | Discards working tree changes |
| `git reset --hard` | Discards commits and working tree |
| `git reset --merge` | Discards merge state |
| `git reset --keep` | Discards working tree selectively |
| `git clean` | Deletes untracked files |
| `git checkout -- <path>` | Restores file from index |
| `git switch -C` | Force-overwrites branch |
| `git branch -D` | Force-deletes branch |
| `git push` | Pushes to remote (all forms) |

Safe read-only commands pass through without restriction: `status`, `diff`, `log`, `show`, `grep`, `ls-files`, `rev-parse`, `branch --show-current`, etc.

## Shell Command Guard

The shell command wrapper (`scripts/safe_cmd.sh`) blocks destructive non-git commands that target protected paths.

### Protected Paths

Directories: `data/`, `config/`, `.git/`, `scripts/`, `memory/`, `core/`, `knowledge/`, `utils/`, `models/`, `gui/`, `eval/`, `integrations/`, `processing/`, `docs/`, `tests/`, `conversation_logs/`

Files: `main.py`, `CLAUDE.md`, `requirements.txt`, `.env`, `pytest.ini`, `daemon.spec`

### Blocked Shell Commands

**Always blocked** (even with unlock — catastrophic, no legitimate agent use case):

| Command | Reason |
|---------|--------|
| `rm -rf /` | Wipes entire filesystem |
| `rm -rf .` | Wipes entire repo |
| `rm -rf ~` | Wipes home directory |
| `rm -rf *` | Wipes current directory contents |

**Blocked by default** (unlockable via same mechanisms as git guard):

| Command | When blocked |
|---------|-------------|
| `rm -rf <path>` | Target is a protected directory or inside one |
| `rm <file>` | Target is a protected root-level file or inside a protected dir |
| `mv <source> <dest>` | Source is a protected path |
| `rmdir <dir>` | Target is a protected directory |
| `chmod 000` / `chmod -R` | Target is a protected path (restrictive modes or recursive) |
| `truncate <file>` | Target is a protected file |
| `find <path> -delete` | Search path is protected AND has `-delete` or `-exec rm` |

### Usage

```bash
bash scripts/safe_cmd.sh rm -rf build/     # safe — passes through
bash scripts/safe_cmd.sh rm -rf data/      # BLOCKED — protected path
bash scripts/safe_cmd.sh mv config/ /tmp/  # BLOCKED — protected path
bash scripts/safe_cmd.sh chmod 000 main.py # BLOCKED — restrictive mode on protected file
```

Non-recognized commands (ls, cat, echo, cp, python, etc.) pass through without restriction.

## Python Filesystem Guard

The Python filesystem guard (`utils/python_fs_guard.py`) protects critical repo paths from in-process Python delete/move/replace operations during agentic tool dispatch.

### How It Works

At startup, `main.py` activates monkey-patches on destructive filesystem functions. When the agentic search controller dispatches tool calls, it sets an "agent mode" `ContextVar`. Only calls made in agent mode are subject to protection checks.

This means Daemon's own runtime code (profile saves, daily note generation, migration scripts) is **never blocked** — only agent-originated operations.

### Protected Operations

| Patched function | Also covers |
|---|---|
| `os.remove()` | direct calls |
| `os.unlink()` | `pathlib.Path.unlink()` |
| `os.rmdir()` | `pathlib.Path.rmdir()` |
| `os.rename()` | `pathlib.Path.rename()` — checks both source AND destination |
| `os.replace()` | `pathlib.Path.replace()` — checks both source AND destination |
| `shutil.rmtree()` | recursive directory removal |
| `shutil.move()` | checks both source AND destination |
| `shutil.copyfile()` | destination-only check (reading source is safe) |
| `shutil.copy()` | destination-only check (reading source is safe) |
| `shutil.copy2()` | destination-only check (reading source is safe) |

### What This Does NOT Protect

This guard prevents common Python delete/move/replace/copy-overwrite operations during in-process agentic tool dispatch. It does **not** protect:

- Arbitrary file writes: `open("protected.py", "w")`, `Path.write_text()`, `Path.write_bytes()`
- Separate Python interpreters that don't import/activate this guard and don't have `scripts/bin/` on PYTHONPATH

### Subprocess Guard

`scripts/bin/usercustomize.py` auto-activates the Python filesystem guard in child Python interpreters when `scripts/bin/` is on `PYTHONPATH` (set by `scripts/activate_guards.sh`). This ensures `subprocess.run(["python", "-c", "..."])` inherits protection without explicit imports.

- Skipped automatically during pytest/coverage runs to avoid test interference
- Can be disabled with the `DISABLE_FS_GUARD=1` environment variable

### Unlock

Same mechanisms as the shell guard:
- `ALLOW_DESTRUCTIVE_OPS=1` environment variable
- `.agent_allow_destructive_once` lockfile

Always-blocked targets (`.`, `..`, `/`, `~`, `*`, repo root) cannot be overridden even with unlock.

## Unlock Mechanisms

When a destructive command is genuinely needed:

**Option 1: Environment variable (per-command)**
```bash
ALLOW_DESTRUCTIVE_OPS=1 scripts/safe_git.sh restore somefile
ALLOW_DESTRUCTIVE_OPS=1 scripts/safe_cmd.sh rm -rf data/old_backup
```

**Option 2: One-shot lockfile (consumed after one use)**
```bash
touch .agent_allow_destructive_once
scripts/safe_git.sh restore somefile
# lockfile is automatically deleted after use
```

Note: The "always blocked" tier (`rm -rf /`, `rm -rf .`, etc.) cannot be overridden even with unlock.

## Emergency Recovery

If an agent has already caused damage:

1. **Stop.** Do not run more commands.
2. Run the audit: `bash scripts/agent_session_audit.sh`
3. Inspect the latest snapshot: `ls .agent_snapshots/`
4. Recover unstaged changes: `git apply .agent_snapshots/<timestamp>/diff.patch`
5. Recover staged changes: `git apply .agent_snapshots/<timestamp>/cached.diff`
6. Recover untracked files: `tar xzf .agent_snapshots/<timestamp>/untracked_files.tar.gz`
7. Compare manifests: `python -m utils.fs_snapshot diff .agent_snapshots/<timestamp>/manifest.json <current_manifest>`

## Components

| File | Purpose |
|------|---------|
| `utils/fs_snapshot.py` | Filesystem manifest creation, diffing, CLI |
| `utils/destructive_op_guard.py` | Git command classifier (Python API) |
| `utils/shell_cmd_guard.py` | Shell command classifier (rm, mv, chmod, etc.) |
| `utils/python_fs_guard.py` | Python filesystem guard (os.remove, shutil.rmtree, etc.) |
| `scripts/agent_session_start.sh` | Pre-agent snapshot script |
| `scripts/agent_session_audit.sh` | Post-agent audit script |
| `scripts/safe_git.sh` | Safe git wrapper with destructive command blocking |
| `scripts/safe_cmd.sh` | Safe shell command wrapper (non-git destructive ops) |
| `tests/unit/test_fs_snapshot.py` | 40 tests for manifest utility |
| `tests/unit/test_destructive_op_guard.py` | 52 tests for git command classifier |
| `tests/unit/test_shell_cmd_guard.py` | 127 tests for shell command classifier |
| `scripts/bin/usercustomize.py` | Subprocess guard: auto-activates python_fs_guard in child interpreters |
| `tests/unit/test_python_fs_guard.py` | 85 tests for Python filesystem guard |

## Known Limitations

- Shell wrappers only work if the agent uses them. For enforcement, configure PATH so `rm`/`mv` etc. resolve to `scripts/bin/` wrappers (see `scripts/activate_guards.sh`).
- Python-level calls (`os.remove()`, `shutil.rmtree()`, `Path.unlink()`, etc.) are guarded by `utils/python_fs_guard.py` during agent tool execution. `subprocess.run(["rm", ...])` is not intercepted by the Python guard (use the shell PATH wrappers for that).
- Arbitrary file writes (`open("x", "w")`, `Path.write_text()`) are not yet guarded.
- Child Python processes inherit the guard when `scripts/bin/` is on PYTHONPATH (via `usercustomize.py`), but ctypes-level and kernel-level bypasses remain unguarded.
- This is defense-in-depth, not a sandbox. The goal is to make destructive actions harder, sessions auditable, and accidental loss recoverable.

## Configuration

- Snapshot directory: `.agent_snapshots/` (gitignored)
- One-shot lockfile: `.agent_allow_destructive_once` (gitignored)
- Max snapshots retained: 10
- Max untracked file size for archival: 25 MB
- Manifest excludes: `.git/`, `__pycache__/`, `.venv/`, `venv/`, `node_modules/`, `RECOVERY_*`, `.agent_snapshots/`, `*.pyc`, `*.rpm`
