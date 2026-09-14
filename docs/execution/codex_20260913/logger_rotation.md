# Conversation logger rotation repair

## Scope and ownership check

Refreshed the exact Plan 2 status command before editing:

```text
git -C /home/lukeh/daemon_exec/generalization status --porcelain --untracked-files=all
```

At that snapshot, Plan 2 had many active files but did not claim `utils/conversation_logger.py`. The updated plan's ownership contract still assigns the class catalog, policy, hooks, workflow, and class-guard tests to the class-guard owner; this batch leaves those files byte-identical and supplies only an unapplied integration patch.

## Evidence and change

Before editing, I loaded `git show HEAD:utils/conversation_logger.py` into memory and ran it in a bounded child process with a temporary log directory and forced text rollover (`max_file_size_mb=0`). The original call timed out at two seconds, reproducing the deadlock without touching repository or user data. The call chain is `log_interaction` (holds `self.lock`) → `_check_rotation` → `_write_session_header` (tries to acquire that same non-reentrant lock).

`utils/conversation_logger.py` now routes header writing through `_write_session_header_unlocked`, whose contract requires the caller to hold the lock. The constructor uses the locking wrapper; rotation, already inside `log_interaction`'s lock, uses the unlocked helper. The rotation scan now checks `self._get_log_filename(index)`, so JSONL mode checks its own extension and skips existing part files.

The new `tests/unit/test_conversation_logger_rotation_contract.py` imports the deployed module by file path in a child process and bounds the whole scenario with an eight-second timeout. It covers multiple text and JSONL parts, checks each entry stays in one part, and exercises concurrent serialized writes. The parent ran this regression under the 512 MB scope: one test passed in 0.050 seconds.

## Class and integration proposal

The in-memory catalog patch adds BC-79, “Same-thread re-entry into a non-reentrant lock,” with status `partial`: it describes the lock mechanism and closes this logger's forced-rotation path, but does not claim project-wide lock analysis. The same patch adds the JSONL part-suffix incident to BC-16, “Related constants drift / one constant, two purposes.”

`logger_guard_integration.patch` proposes adding the new test to the configured repo-wide guard list, its exact expected-list fixture, the pre-push guard array, and the required CI guard command. It also changes the affected hook/workflow “five guards” wording to generic required-guard wording. The patch is not applied. It does not alter the class-guard harness's pinned 311-test count or its assertions; the existing policy test already checks exact expected-list parity, so the fixture addition is included.

The unified patch passes `git apply --check`. In-memory candidate validation passed for the catalog index/entry, the six-item policy and expected fixture, and the exact hook/workflow enforcement validators supplied with the six candidate guard paths. The six protected source files remain byte-identical to `HEAD`.

The patch should be applied/reviewed by the class-guard owner with the catalog and enforcement wiring. The candidate also updates the current guard lists in `docs/TEST_LANES.md` and `docs/DEVELOPMENT_WORKFLOW.md`; historical five-guard run counts are preserved. Those documents remain unchanged on disk.

## Handoff

Run centrally, with no other pytest process active:

```bash
systemd-run --user --scope -p MemoryMax=512M python -m unittest discover -s tests/unit -p test_conversation_logger_rotation_contract.py
```

The source/test patch is review-ready and its bounded regression passed. No commit or push was made.
