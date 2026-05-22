# Agent Guard - Architecture & Skeleton

**Purpose**: Compressed architectural overview of `agent_guard.py` — a standalone
agent safety guard that protects repository paths from AI agent filesystem
operations, and classifies git/shell commands as safe or destructive.

**Last Updated**: 2026-05-22

---

## 1. Architecture Overview

```
AGENT TOOL DISPATCH
    |
    v
[agent_mode() context manager]          <-- ContextVar, async-safe
    |
    v
[Guarded Python FS calls]               <-- 10 monkey-patched functions
    |                                        os.remove, os.unlink, os.rmdir,
    |                                        os.rename, os.replace, shutil.rmtree,
    |                                        shutil.move, shutil.copyfile,
    |                                        shutil.copy, shutil.copy2
    v
[_check_and_maybe_block()]              <-- Core guard decision logic
    |
    +--> always-blocked? (., .., /, ~, *)  --> PermissionError (no unlock)
    +--> resolve to repo-relative path
    +--> outside repo?                     --> pass through
    +--> not protected?                    --> pass through
    +--> unlock in effect?                 --> warn + pass through
    +--> BLOCKED                           --> PermissionError

STANDALONE CLASSIFIERS (no activation required)
    |
    +--> classify_git_command(args)        <-- Stateless git arg classifier
    |       always-destructive: push, restore, clean
    |       conditional: reset --hard, branch -D, switch -C
    |       special: checkout (-- = destructive, -b = safe)
    |
    +--> classify_shell_command(args)      <-- Stateless shell cmd classifier
            rm, mv, rmdir, chmod, truncate, find
            path resolution via PurePosixPath (no fs access)
            two severity tiers: "always" vs "protected"
```

---

## 2. Module Structure

```
agent_guard.py (single file, ~940 lines)
|
+-- Constants
|   ALWAYS_BLOCKED_TARGETS: frozenset  = {".", "..", "/", "~", "*"}
|
+-- Module State (set by activate())
|   _active: bool
|   _repo_root: Path | None
|   _protected_dirs: frozenset[str]
|   _protected_files: frozenset[str]
|   _lockfile_name: str
|   _originals: dict[str, Any]           # saved stdlib function pointers
|   _agent_mode: ContextVar[bool]        # async-safe per-context flag
|
+-- Section 1: Git Command Classifier
|   classify_git_command(args) -> dict    # {subcmd, destructive, reason}
|   is_destructive_git(args) -> bool
|
+-- Section 2: Unlock Check
|   unlock_allowed(env, root, lockfile_name) -> bool
|
+-- Section 3: Path Resolution Helpers
|   _resolve_target(target, repo_root) -> str | None
|   _is_protected(rel_path, dirs, files) -> bool
|
+-- Section 4: Shell Command Classifiers
|   classify_shell_command(args, repo_root, dirs, files) -> dict
|   is_destructive_shell(args, repo_root, dirs, files) -> bool
|   _classify_rm(args, root, dirs, files) -> dict
|   _classify_mv(args, root, dirs, files) -> dict
|   _classify_rmdir(args, root, dirs, files) -> dict
|   _classify_chmod(args, root, dirs, files) -> dict
|   _classify_truncate(args, root, dirs, files) -> dict
|   _classify_find(args, root, dirs, files) -> dict
|   _parse_rm_flags(args) -> (bool, bool)
|   _extract_targets(args) -> list[str]
|
+-- Section 5: Python FS Guard
    activate(repo_root, dirs, files, lockfile) -> None
    deactivate() -> None
    is_active() -> bool
    set_agent_mode(active) -> None
    agent_mode() -> ContextManager          # ContextVar token-based
    _check_and_maybe_block(op, target) -> None   # core decision
    _resolve_to_repo_relative(target) -> str | None
    _is_protected_path(rel_path) -> bool    # uses stored config
    _is_always_blocked(raw, resolved) -> bool
    _guarded_remove(path, *a, **kw)         # 10 wrapper functions
    _guarded_unlink(...)
    _guarded_rmdir(...)
    _guarded_rename(src, dst, ...)
    _guarded_replace(src, dst, ...)
    _guarded_rmtree(...)
    _guarded_move(src, dst, ...)
    _guarded_copyfile(src, dst, ...)
    _guarded_copy(src, dst, ...)
    _guarded_copy2(src, dst, ...)
```

---

## 3. Guard Decision Flow

The `_check_and_maybe_block()` function implements a 9-step decision cascade:

```
Step  Check                          Result
----  ----                           ------
 1    Guard not active?              pass through
 2    Not in agent_mode()?           pass through
 3    Raw target always-blocked?     PermissionError (NO unlock)
 4    Resolve to repo-relative       --
 5    Resolved always-blocked?       PermissionError (NO unlock)
 6    Outside repo root?             pass through
 7    Not a protected path?          pass through
 8    unlock_allowed()?              warn + pass through
 9    Otherwise                      PermissionError (with unlock instructions)
```

**Always-blocked** targets (`"."`, `".."`, `"/"`, `"~"`, `"*"`) and the repo root
itself can never be unlocked — they represent catastrophic operations with no
legitimate agent use case.

**Protected paths** (user-configured dirs and files) can be unlocked via:
- `ALLOW_DESTRUCTIVE_OPS=1` environment variable
- One-shot lockfile (default: `.agent_allow_destructive_once` in repo root)

---

## 4. Shell Command Classification

Each command has a dedicated classifier that returns:
```python
{"command": str, "destructive": bool, "severity": str|None, "reason": str|None, "targets": list[str]}
```

| Command    | What triggers "destructive"                                     | Severity    |
|------------|----------------------------------------------------------------|-------------|
| `rm`       | Any target in protected dir/file, or `-r` on always-blocked    | protected/always |
| `mv`       | Source is a protected path                                      | protected/always |
| `rmdir`    | Target is a protected dir                                       | protected/always |
| `chmod`    | Restrictive mode (000/0100/0200) or `-R` on protected path     | protected   |
| `truncate` | Target is a protected path                                      | protected   |
| `find`     | `-delete` or `-exec rm` with search path in protected dir       | protected/always |

Path resolution uses `PurePosixPath` (no filesystem access) for testability.
Paths outside the repo root return `None` and are ignored.

---

## 5. Git Command Classification

Stateless, no config needed. Returns:
```python
{"subcmd": str, "destructive": bool, "reason": str|None}
```

| Category                  | Commands / Flags                                |
|---------------------------|-------------------------------------------------|
| Always destructive        | `push`, `restore`, `clean`                      |
| Conditionally destructive | `reset --hard/--merge/--keep`, `branch -D`, `switch -C` |
| Special: checkout         | `--` separator = destructive; `-b`/`-B` = safe   |
| Everything else           | Safe                                             |

---

## 6. Key Design Decisions

### Why ContextVar for agent_mode?

`asyncio.gather()` copies ContextVar state to child tasks, so each concurrent
agent tool dispatch gets its own independent copy of the flag. Thread-local
storage would not work here — all async tasks share one thread.

### Why PurePosixPath in shell classifiers?

Pure path math avoids filesystem access, making the classifiers deterministic
and testable without creating real directory structures. The Python FS guard
uses `Path.resolve()` instead (needs real paths for monkey-patched calls).

### Why two-tier blocking?

Always-blocked targets represent universally catastrophic operations (delete
the entire filesystem, the repo root, etc.). These must never be unlockable,
even by accident. Protected paths are project-specific and may legitimately
need override during migrations, cleanup scripts, etc.

### Why monkey-patching?

Agent tools that call `os.remove()` or `shutil.rmtree()` in-process bypass
any shell wrapper. Monkey-patching catches these calls at the Python API
level. The guard is transparent — it only intervenes in agent_mode() contexts,
and deactivate() fully restores the originals.

---

## 7. Integration Pattern

```python
# At application startup (before agent code loads):
from agent_guard import activate, agent_mode

activate(
    repo_root=Path(__file__).parent,
    protected_dirs={".git", "src", "config", "data", "tests"},
    protected_files={".env", "main.py", "pyproject.toml"},
)

# In your agent dispatch loop:
async def dispatch_agent_tool(tool_name, tool_args):
    with agent_mode():
        return await execute_tool(tool_name, tool_args)

# For shell command pre-screening:
from agent_guard import classify_shell_command

def run_shell(cmd_args):
    result = classify_shell_command(cmd_args, repo_root=".")
    if result["destructive"]:
        raise PermissionError(result["reason"])
    subprocess.run(cmd_args)
```

---

## 8. Invariants

1. **No false negatives on always-blocked**: Operations on `.`, `..`, `/`, `~`, `*`,
   and the repo root itself are always blocked in agent mode, regardless of config.
2. **Transparent outside agent_mode**: All 10 patched functions pass through with
   zero overhead when `_agent_mode.get(False)` is `False`.
3. **Idempotent activation**: Calling `activate()` twice is a no-op. Calling
   `deactivate()` twice is a no-op.
4. **No data loss on deactivation**: `_originals` dict preserves the real stdlib
   functions and restores them exactly.
5. **Stateless classifiers**: `classify_git_command()` and `classify_shell_command()`
   work without `activate()` — they accept all config as parameters.
