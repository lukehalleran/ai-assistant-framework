#data/pipeline/extract_wikipedia_articles.py
"""
Wikipedia XML dump extractor — filters to namespace 0 (main articles) only.

Skips: Talk, User, Wikipedia, File, Template, Category, Draft, etc.
Also skips redirect pages and stubs under 100 chars.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from lxml import etree
import logging

logger = logging.getLogger(__name__)

# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)
XML_FILE = os.path.join(BASE_DIR, "wiki_data", "enwiki-latest-pages-articles.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "wiki_articles")
MAX_WORKERS = 8

# Minimum article text length (after stripping) to be considered a real article
MIN_TEXT_LENGTH = 100

# === FUNCTIONS ===

def _is_article(ns_elem, text: str) -> bool:
    """
    Check if a page is a real main-namespace article.

    Filters:
        1. Namespace must be 0 (main article space)
        2. Text must not be a redirect (#REDIRECT [[...]])
        3. Text must be longer than MIN_TEXT_LENGTH
    """
    # Namespace filter: only ns=0 (main articles)
    if ns_elem is None or (ns_elem.text or "").strip() != "0":
        return False

    stripped = text.strip()

    # Skip redirects
    if stripped.upper().startswith("#REDIRECT"):
        return False

    # Skip stubs / empty pages
    if len(stripped) < MIN_TEXT_LENGTH:
        return False

    return True


def clean_text(text):
    """Basic cleanup for article text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_article(title, text):
    """Save a single article to disk."""
    safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)
    filename = f"{safe_title[:100]}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f_out:
        f_out.write(clean_text(text))


def process_wiki_dump(xml_file):
    """Extract and save articles from the Wikipedia dump (namespace 0 only)."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    context = etree.iterparse(xml_file, events=("end",), tag="{*}page")
    article_count = 0
    skipped = {"non_article_ns": 0, "redirect": 0, "too_short": 0}

    print("[INFO] Extraction started (filtering to namespace 0 articles only)...")
    first_titles = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for event, elem in context:
            title_elem = elem.find("./{*}title")
            ns_elem = elem.find("./{*}ns")
            text_elem = elem.find(".//{*}text")

            if title_elem is not None and text_elem is not None:
                title = title_elem.text or "Untitled"
                text = text_elem.text or ""

                if not _is_article(ns_elem, text):
                    # Track skip reasons for reporting
                    ns_val = (ns_elem.text or "").strip() if ns_elem is not None else "?"
                    if ns_val != "0":
                        skipped["non_article_ns"] += 1
                    elif text.strip().upper().startswith("#REDIRECT"):
                        skipped["redirect"] += 1
                    else:
                        skipped["too_short"] += 1
                else:
                    futures.append(executor.submit(save_article, title, text))
                    article_count += 1

                    if len(first_titles) < 5:
                        first_titles.append(title)

                    if article_count % 10000 == 0:
                        total_skipped = sum(skipped.values())
                        print(f"[PROGRESS] {article_count:,} articles extracted, "
                              f"{total_skipped:,} skipped "
                              f"(ns={skipped['non_article_ns']:,}, "
                              f"redir={skipped['redirect']:,}, "
                              f"short={skipped['too_short']:,})")

            elem.clear()

    for future in futures:
        future.result()

    total_skipped = sum(skipped.values())
    print(f"\n[DONE] Extraction complete.")
    print(f"  Articles extracted: {article_count:,}")
    print(f"  Pages skipped:      {total_skipped:,}")
    print(f"    Non-article NS:   {skipped['non_article_ns']:,}")
    print(f"    Redirects:        {skipped['redirect']:,}")
    print(f"    Too short:        {skipped['too_short']:,}")
    print(f"  First titles: {first_titles}")


def stream_extract_articles(xml_file):
    """
    Generator to stream main-namespace Wikipedia articles.

    Yields dicts with keys: title, text, page_id
    Filters: namespace 0 only, no redirects, min length 100 chars.
    """
    context = etree.iterparse(xml_file, events=("end",), tag="{*}page")
    article_count = 0
    skipped_count = 0

    for event, elem in context:
        title_elem = elem.find("./{*}title")
        ns_elem = elem.find("./{*}ns")
        id_elem = elem.find("./{*}id")
        text_elem = elem.find(".//{*}text")

        if title_elem is not None and text_elem is not None:
            title = title_elem.text or "Untitled"
            text = text_elem.text or ""
            page_id = (id_elem.text or "0") if id_elem is not None else "0"

            if _is_article(ns_elem, text):
                article_count += 1
                if article_count % 100000 == 0:
                    print(f"[STREAM] {article_count:,} articles yielded, "
                          f"{skipped_count:,} non-articles skipped")

                yield {
                    "title": title,
                    "text": text,
                    "page_id": page_id,
                }
            else:
                skipped_count += 1

        elem.clear()

    print(f"[STREAM COMPLETE] {article_count:,} articles yielded, "
          f"{skipped_count:,} non-articles skipped")


# === MAIN ===

if __name__ == "__main__":
    print("[INIT] Starting Wikipedia extraction...")
    process_wiki_dump(XML_FILE)
