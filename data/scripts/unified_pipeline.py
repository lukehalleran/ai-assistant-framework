# unified_pipeline.py
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("unified_pipeline.py is alive")
import argparse
import os
import subprocess
import urllib.request
from pathlib import Path
import sys
import json
from typing import Dict, List
import logging

# Import both pipelines
# Make sure these are callable as functions
from embed_wiki_chunks_to_parquet import run_embedding_pipeline
from semantic_chunker import SemanticWikiChunker

# Default paths
DEFAULT_WIKI_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
WIKI_ZIP_PATH = Path("wiki_data/enwiki-latest-pages-articles.xml.bz2")
EXTRACTED_PATH = Path("wiki_data/enwiki-latest-pages-articles.xml")
CHUNKS_DIR = Path("wiki_chunks") # This is for basic chunking
SEMANTIC_CHUNKS_DIR = Path("semantic_chunks") # This is where semantic chunker outputs

# Define TEST_MODE globally or pass it
TEST_MODE = os.environ.get("TEST_MODE") == "1" # Add this line

def download_wikipedia_dump(url, output_path):
    """Download Wikipedia dump from URL with progress reporting."""
    logger.debug(f"🌐 Downloading Wikipedia dump from: {url}")
    logger.debug(f"📁 Saving to: {output_path}")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\r⬇️  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=download_progress)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def unzip_wikipedia_dump(zip_path):
    """Unzip bz2 compressed Wikipedia dump."""
    print("📦 Unzipping Wikipedia dump...", flush=True)

    if not zip_path.exists():
        print(f"❌ File not found: {zip_path}")
        return False

    # Use -k to keep the original file, -f to force overwrite if exists
    result = subprocess.run(
        ["bzip2", "-dkf", str(zip_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Unzipping complete.")
        return True
    else:
        print("❌ Unzip failed:", result.stderr)
        return False

def run_basic_extractor():
    """Run the original article extraction (500-article chunks)."""
    print("🔍 Running BASIC article extraction (500-article chunks)...", flush=True)

    if not EXTRACTED_PATH.exists():
        print(f"❌ Extracted XML not found at {EXTRACTED_PATH}")
        return False

    result = subprocess.run(
        ["python", "extract_wikipedia_articles.py"], # Assumes extract_wikipedia_articles.py exists
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("[Extractor STDERR]", result.stderr)

    return result.returncode == 0

def run_semantic_extractor(xml_path=None, chunk_size=1000, chunk_overlap=200, min_chunk_size=100, max_articles=None):
    global TEST_MODE
    if TEST_MODE:
        print("🛑 TEST_MODE is on — skipping semantic extractor!", flush=True)
        return True

    print("🧩 Running SEMANTIC extraction and chunking...", flush=True)

    xml_path = xml_path or EXTRACTED_PATH
    if not xml_path.exists():
        print(f"❌ Extracted XML not found at {xml_path}")
        return False

    try:
        chunker = SemanticWikiChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

        chunker.stream_process_wikipedia(
            str(xml_path),
            str(SEMANTIC_CHUNKS_DIR),
            batch_size=1000,
            max_articles=max_articles  # <<< Pass it here!
        )

        return True
    except Exception as e:
        print(f"❌ Semantic chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Remove convert_jsonl_to_text_format and embed_semantic_chunks
# We will directly use embed_wiki_chunks_to_parquet with the JSONL files.

def main():
    parser = argparse.ArgumentParser(
        description="Daemon Wikipedia Embedding Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --download --semantic
  python pipeline.py --download
  python pipeline.py --compressed wiki_data/enwiki.xml.bz2 --semantic
  python pipeline.py --extracted wiki_data/enwiki.xml --semantic
  python pipeline.py --extracted data.xml --test --semantic
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--download", action="store_true", help="Download Wikipedia dump")
    input_group.add_argument("--compressed", type=Path, metavar="PATH", help="Use existing .bz2 file")
    input_group.add_argument("--extracted", type=Path, metavar="PATH", help="Use extracted XML")

    parser.add_argument("--semantic", action="store_true", help="Use semantic chunking")
    parser.add_argument("--url", type=str, default=DEFAULT_WIKI_URL, help="Download URL")
    parser.add_argument("--source", type=str, default="wikipedia.com", help="Source label (currently not directly used in embedding)")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip chunking step")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Semantic chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--max-articles", type=int, default=None, help="Limit number of articles for semantic chunker")


    args = parser.parse_args()

    # Set TEST_MODE environment variable
    if args.test:
        os.environ["TEST_MODE"] = "1"
    else:
        os.environ["TEST_MODE"] = "0" # Explicitly set to "0" if not test
    global TEST_MODE # Update global TEST_MODE variable
    TEST_MODE = os.environ.get("TEST_MODE") == "1"

    os.environ["NUM_CONSUMER_THREADS"] = "1" # Or allow this to be an argument

    extracted_xml_path = EXTRACTED_PATH

    if args.download:
        if not download_wikipedia_dump(args.url, WIKI_ZIP_PATH):
            sys.exit(1)
        if not unzip_wikipedia_dump(WIKI_ZIP_PATH):
            sys.exit(1)

    elif args.compressed:
        if args.compressed != WIKI_ZIP_PATH:
            print(f"📁 Using compressed file: {args.compressed}")
        if not unzip_wikipedia_dump(args.compressed):
            sys.exit(1)

    elif args.extracted:
        extracted_xml_path = args.extracted
        print(f"📁 Using extracted XML: {extracted_xml_path}")
        if not extracted_xml_path.exists():
            print(f"❌ File not found: {extracted_xml_path}")
            sys.exit(1)
    # Extraction phase
    if not args.skip_extraction:
        if args.semantic:
            # Pass chunking arguments to semantic extractor
            if not run_semantic_extractor(
                extracted_xml_path,
                args.chunk_size,
                args.chunk_overlap,
                max_articles=args.max_articles  # <<< ADD max_articles here too!
            ):
                print("❌ Semantic extraction failed!")
                sys.exit(1)
        else:
            if args.extracted is None:
                if not run_basic_extractor():
                    print("❌ Basic extraction failed!")
                    sys.exit(1)

        # Embedding phase
        try:
            run_embedding_pipeline()
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            sys.exit(1)

        print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
