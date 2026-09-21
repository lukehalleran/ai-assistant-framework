# Executor brief — template for a cheap subagent batch

Copy, fill the `{…}` fields, paste as the subagent's whole prompt. One batch = one owner group of
files, one kind of change, ≤8 files / ≤40 records / ≤300 changed lines unless the plan says otherwise.
Every rule below exists because a batch broke it once (`docs/DEVELOPMENT_WORKFLOW.md` §8).

```
You are executing ONE batch of {PLAN PATH}. Read {PLAN SECTIONS} first, fully.
BATCH: {ID}    CHANGE KIND: {HOIST | MARK | CONVERT | refactor | test-only | …}
FILES (yours, exactly): {list}
Other agents may be editing OTHER files in the same tree. Never touch a file outside your list; never
`git stash` (even scoped), `git checkout`, `git restore`, `git reset`, `git clean`; never revert someone
else's change. Never write to ~/Daemon_v1, ~/daemon_checkpoints, any data/ directory, or class-guard files
(scripts/check_bug_classes.py, scripts/bug_class_guards/, tests/bug_class_guards/, hooks/, .github/,
config/bug_class_*). Use /usr/bin/rm, never rm. Do not commit or push. Do not start the Daemon.
Do not edit tests except the ONE edit the plan pre-authorizes: {none | the ceiling constant in …}.

Working tree: {clone path} (branch {name}). Artifacts: {runs dir}/{ID}/.
Every python command, from the clone root:
  export PY="env -u PYTHONPATH DISABLE_FS_GUARD=1 DAEMON_TEST_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python -s"
  cd {clone} && $PY -c "import utils.safe_json as m; assert m.__file__.startswith('$PWD'), m.__file__; print('clone code OK')"
Print that proof line first; if it is missing, STOP (the login shell preloads the LIVE repo's utils otherwise).
pytest ALWAYS runs as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=512M $PY -m pytest -q -p no:cacheprovider <files>
(never add PYTEST_DISABLE_PLUGIN_AUTOLOAD — it disables pytest-asyncio and every async test errors),
after waiting for any real pytest: while pgrep -f "python(3)?( -s)? -m pytest" >/dev/null; do sleep 20; done
(give up after 30 min → STOP). Before-baselines come from the pristine clone {base clone}, never from the shared tree.

Steps:
1. {measure / inventory the records you own; paste the count}
2. {apply the plan's edit pattern to EXACTLY those records; the plan's `edit` field is the edit — never re-derive}
3. Checks (paste every output with counts): {plan-specific mechanical check}; $PY -m ruff check <files>;
   $PY -c "import <module>" for every touched module; the tests the plan names + the five repo-wide guards
   (tests/unit/test_no_git_state_in_tests.py tests/unit/test_ordered_slice_guard.py
   tests/unit/test_budget_meters_rendered_sections.py tests/unit/test_tool_wiring_parity.py
   tests/unit/test_model_capability_wiring.py); $PY scripts/check_bug_classes.py scan --root . ("bug-class scan: OK";
   a disposition_source_changed line for a file you touched = you touched a DEBT file = STOP); git status --short
   (only your files among YOUR changes).
4. Handoff {runs dir}/{ID}/handoff.md in the block format STATE / CLASS / ACTIONS / PLANNED / CONTINGENCY / WHY /
   OWNER (docs/DEVELOPMENT_WORKFLOW.md §5) + files.txt (one path per line) + commit_message.txt.
5. CANDIDATE BUG CLASSES: a recurring MECHANISM not in docs/BUG_CLASSES.md (read its index) with ≥2 file:line
   instances goes in the handoff. Never edit the catalog yourself.

STOP conditions: any test fails; a debt file would be touched; an edit needs a form the plan does not give;
you would need a file outside your list; free -g < 3 GB; {plan-specific stops}. On a STOP: write the handoff with
what you have and end your final message with "ESCALATION: <one sentence>". Never improvise a workaround.
Final message: the STATE/ACTIONS/CONTINGENCY blocks (≤60 lines) + the handoff path.
```

Referee checklist for the frontier that receives the handoff: read the diff, re-run one check yourself, run the
BC-58 sibling grep on every touched function, confirm the handoff's counts against `git diff --stat`, and write the
combined commit message only after `git diff --stat` matches the file list.
