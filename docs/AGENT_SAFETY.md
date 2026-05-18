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

## Unlock Mechanisms

When a destructive command is genuinely needed:

**Option 1: Environment variable (per-command)**
```bash
ALLOW_DESTRUCTIVE_OPS=1 scripts/safe_git.sh restore somefile
```

**Option 2: One-shot lockfile (consumed after one use)**
```bash
touch .agent_allow_destructive_once
scripts/safe_git.sh restore somefile
# lockfile is automatically deleted after use
```

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
| `scripts/agent_session_start.sh` | Pre-agent snapshot script |
| `scripts/agent_session_audit.sh` | Post-agent audit script |
| `scripts/safe_git.sh` | Safe git wrapper with destructive command blocking |
| `tests/unit/test_fs_snapshot.py` | 40 tests for manifest utility |
| `tests/unit/test_destructive_op_guard.py` | 52 tests for command classifier |

## Known Limitations

- The safe git wrapper only works if the agent uses it. An agent that calls `git` directly bypasses it.
- For stronger enforcement, configure the agent's PATH so `git` resolves to the wrapper.
- This does not intercept arbitrary shell commands (`rm -rf`, etc.) or Python calls (`shutil.rmtree`).
- This is defense-in-depth, not a sandbox. The goal is to make destructive actions harder, sessions auditable, and accidental loss recoverable.

## Configuration

- Snapshot directory: `.agent_snapshots/` (gitignored)
- One-shot lockfile: `.agent_allow_destructive_once` (gitignored)
- Max snapshots retained: 10
- Max untracked file size for archival: 25 MB
- Manifest excludes: `.git/`, `__pycache__/`, `.venv/`, `venv/`, `node_modules/`, `RECOVERY_*`, `.agent_snapshots/`, `*.pyc`, `*.rpm`
