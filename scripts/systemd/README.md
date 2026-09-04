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
