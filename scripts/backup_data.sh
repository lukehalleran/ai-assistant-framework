#!/bin/bash
# Daily backup of Daemon critical data (conversations, ChromaDB, profiles, graph).
# Keeps the last 7 backups, deletes older ones.
#
# Install as daily cron:
#   crontab -e
#   0 3 * * * /home/lukeh/Daemon_v1/scripts/backup_data.sh
#
# Or run manually:
#   bash scripts/backup_data.sh

set -euo pipefail

PROJECT_DIR="/home/lukeh/Daemon_v1"
BACKUP_DIR="/home/lukeh/daemon_backups"
KEEP_DAYS=7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/daemon_data_${TIMESTAMP}.tar.gz"

mkdir -p "$BACKUP_DIR"

cd "$PROJECT_DIR"

tar czf "$BACKUP_FILE" \
  data/chroma_multi/ \
  data/chroma_db_v4/ \
  data/corpus_v4.json \
  data/corpus/ \
  data/knowledge_graph.json \
  data/entity_aliases.json \
  data/user_profile.json \
  data/claim_index.json \
  data/surfacing_history.json \
  data/benchmark_results.json \
  data/benchmark_history.json \
  2>/dev/null || true

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[$(date)] Backup created: $BACKUP_FILE ($SIZE)"

# Upload encrypted backup to Backblaze B2
RCLONE_REMOTE="b2-daemon-crypt:backups"
echo "[$(date)] Uploading to Backblaze B2 (encrypted)..."
if rclone copy "$BACKUP_FILE" "$RCLONE_REMOTE/" --progress 2>&1; then
  echo "[$(date)] Upload complete."
else
  echo "[$(date)] ERROR: Upload to B2 failed!" >&2
fi

# Prune local backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "daemon_data_*.tar.gz" -mtime +${KEEP_DAYS} -delete 2>/dev/null
REMAINING=$(find "$BACKUP_DIR" -name "daemon_data_*.tar.gz" | wc -l)
echo "[$(date)] Keeping $REMAINING local backups (last ${KEEP_DAYS} days)"

# Prune remote backups older than KEEP_DAYS
rclone delete "$RCLONE_REMOTE/" --min-age "${KEEP_DAYS}d" 2>/dev/null
echo "[$(date)] Pruned remote backups older than ${KEEP_DAYS} days"
