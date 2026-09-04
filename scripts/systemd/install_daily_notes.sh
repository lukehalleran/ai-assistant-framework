#!/usr/bin/env bash
# Install/refresh the Daemon daily-notes systemd USER units from the repo templates.
# Idempotent. Run from the repo root:  bash scripts/systemd/install_daily_notes.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest/daemon-daily-notes.service.d"
cp "$here/daemon-daily-notes.service"        "$dest/"
cp "$here/daemon-daily-notes.timer"          "$dest/"
cp "$here/daemon-daily-notes-failed.service" "$dest/"
cp "$here/daemon-daily-notes.service.d/onfailure.conf" "$dest/daemon-daily-notes.service.d/"
echo "installed to $dest:"; ls -1 "$dest" | grep daily-notes
systemctl --user daemon-reload
systemctl --user enable --now daemon-daily-notes.timer
echo "--- ExecStart now:"; systemctl --user cat daemon-daily-notes.service | grep ExecStart
echo "--- running the job once (idempotent):"
systemctl --user start daemon-daily-notes.service || true
journalctl --user -u daemon-daily-notes.service -n 20 --no-pager
echo "--- next timer run:"; systemctl --user list-timers --no-pager | grep -E "NEXT|daily-notes"
