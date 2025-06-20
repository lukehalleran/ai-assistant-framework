# test_full_pipeline.py

import os
import subprocess
import time

# === CONFIG ===
TEST_TOPICS = [
    "Albert Einstein",
    "World War II",
    "Python (programming language)",
    "Black hole",
    "Artificial intelligence",
    "Machine learning",
    "Quantum mechanics",
    "Philosophy",
    "Internet",
    "Climate change"
]

# === PIPELINE STEPS ===

# 1️⃣ Download Wikipedia articles
print("\n✅ Step 1: Downloading articles...")
with open("test_topics.txt", "w", encoding="utf-8") as f:
    for topic in TEST_TOPICS:
        f.write(topic + "\n")

# Run download_wikipedia.py
subprocess.run(["python", "download_wikipedia.py", "test_topics.txt", "test_wiki_dump"], check=True)

# 2️⃣ Extract + Chunk + Embed using unified_pipeline.py
print("\n✅ Step 2: Running unified_pipeline.py...")
subprocess.run(["python", "unified_pipeline.py", "--input_dir", "test_wiki_dump", "--output_dir", "test_parquet"], check=True)

# 3️⃣ Build FAISS index and run search test
print("\n✅ Step 3: Running search test on new index...")

# Now import your existing search_faiss_with_metadata.py
# (We import at runtime so the index loads AFTER the pipeline runs)
import search_faiss_with_metadata as search_module

# Run test queries
print("\n🎉 Semantic Search Results:\n")
for query in TEST_TOPICS:
    print(f"\n🟢 Query: '{query}'")
    search_results = search_module.semantic_search(query)
    for result in search_results:
        print(f"  Rank {result['rank']}: {result['title']} (Similarity: {result['similarity']:.4f})")
        print(f"    Preview: {result['text']}\n")

print("\n✅ Full pipeline test completed successfully!\n")
