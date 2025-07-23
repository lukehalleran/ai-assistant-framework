import json
from datetime import datetime

CORPUS_PATH = "/home/lukeh/Daemon_RAG_Agent/data/corpus_v4.json"

BACKUP_PATH = CORPUS_PATH + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    corpus = json.load(f)

filtered = [
    entry for entry in corpus
    if "Sunny and 75°F." not in entry.get("response", "") and
       "Hello, how are you?" not in entry.get("query", "")
]

print(f"🧹 Found {len(corpus) - len(filtered)} entries to remove.")

# Backup original
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

# Write cleaned corpus
with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned corpus written to {CORPUS_PATH}. Backup saved as {BACKUP_PATH}.")
