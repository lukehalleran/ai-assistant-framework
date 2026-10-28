# systemd user units for Daemon maintenance jobs

Templates for the timers that run outside the app process. They use `%h`
(the user's home) and assume the repo lives at `~/Daemon_v1` and the project
interpreter is the pyenv 3.11.8 environment (pyproject requires
`python >=3.11,<3.12`). Adjust `WorkingDirectory=` / `ExecStart=` if your
checkout or interpreter differs.

## daily notes (02:00)

```bash
mkdir -p ~/.config/systemd/user/daemon-daily-notes.service.d
cp scripts/systemd/daemon-daily-notes.service  ~/.config/systemd/user/
cp scripts/systemd/daemon-daily-notes.timer    ~/.config/systemd/user/
cp scripts/systemd/daemon-daily-notes-failed.service ~/.config/systemd/user/
cp scripts/systemd/daemon-daily-notes.service.d/onfailure.conf \
   ~/.config/systemd/user/daemon-daily-notes.service.d/
systemctl --user daemon-reload
systemctl --user enable --now daemon-daily-notes.timer
systemctl --user list-timers                       # verify next run
systemctl --user start daemon-daily-notes.service  # run once now
journalctl --user -u daemon-daily-notes.service -n 20
```

The service runs `scripts/daily_note_catchup.py`, a thin entrypoint that
imports only the notes generator and the LLM client (never the orchestrator or
the Chroma store). It exits 1 on a generation error, which triggers the
`OnFailure=` desktop alert; "note already exists" and "skipped" exit 0.

`logs/daily_notes.log` is append-only from the unit; the app's startup log
maintenance rotates it above `log_maintenance.daily_notes_max_mb`.

## weekly tests (Monday 03:30)

Runs the FULL suite with no marker filter (so the slow, semantic and benchmark groups pushes skip are included),
split like pre-push (`tests/unit` under 6G, the rest under 8G), in a dedicated scratch clone
(`~/daemon_exec/weekly_tests_clone`, detached at the live checkout's HEAD). If
`~/Daemon_v1/data/benchmarks_private/retrieval_benchmarks_real.yaml` exists, that one file is copied into the clone
(gitignored there) so the real-data retrieval benchmark runs; `--real-benchmark PATH` overrides the source. It skips
(exit 0) while the Daemon runs, when MemAvailable < 9000 MB, or when another run holds the lock; it refuses (exit 2)
if the scratch clone has local changes. `Persistent=false`: a missed Monday is NOT run at the next boot.

```bash
cp scripts/systemd/daemon-weekly-tests.service        ~/.config/systemd/user/
cp scripts/systemd/daemon-weekly-tests.timer          ~/.config/systemd/user/
cp scripts/systemd/daemon-weekly-tests-failed.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now daemon-weekly-tests.timer
systemctl --user list-timers daemon-weekly-tests.timer   # next run: Mon 03:30
python scripts/weekly_test_run.py --dry-run               # prints the plan, runs nothing
systemctl --user start daemon-weekly-tests.service        # run once now (hours; Daemon must be down)
```

Reports: `logs/weekly_tests/<YYYY-MM-DD>.txt` (counts, failures, `-rs` skip reasons) and one JSON line per run in
`logs/weekly_tests/summary.jsonl`; unit output in `logs/weekly_tests.log`.
