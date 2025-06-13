import os
import re
from concurrent.futures import ThreadPoolExecutor
from lxml import etree

# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)
XML_FILE = os.path.join(BASE_DIR, "wiki_data", "enwiki-latest-pages-articles.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "wiki_articles")
MAX_WORKERS = 8  # Adjust depending on CPU

# === FUNCTIONS ===

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
    """Extract and save articles from the Wikipedia dump."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    context = etree.iterparse(xml_file, events=("end",), tag="{*}page")
    article_count = 0

    print("[INFO] Extraction started...")
    first_titles = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for event, elem in context:
            title_elem = elem.find("./{*}title")
            text_elem = elem.find(".//{*}text")

            if title_elem is not None and text_elem is not None:
                title = title_elem.text or "Untitled"
                text = text_elem.text or ""

                if len(text.strip()) > 100:
                    futures.append(executor.submit(save_article, title, text))
                    article_count += 1

                    if len(first_titles) < 3:
                        first_titles.append(title)

                    if article_count % 100 == 0:
                        print(f"[PROGRESS] {article_count} articles scheduled...")

            elem.clear()

    for future in futures:
        future.result()

    print(f"[DONE] Extraction complete. {article_count} articles saved.")
    print(f"[DEBUG] First few titles extracted: {first_titles}")
def stream_extract_articles(xml_file):
    """Generator to stream Wikipedia articles as XML elements."""
    context = etree.iterparse(xml_file, events=("end",), tag="{*}page")

    for event, elem in context:
        title_elem = elem.find("./{*}title")
        text_elem = elem.find(".//{*}text")

        if title_elem is not None and text_elem is not None:
            title = title_elem.text or "Untitled"
            text = text_elem.text or ""

            yield {
                "title": title,
                "text": text
            }

        elem.clear()

# === MAIN ===

if __name__ == "__main__":
    print("[INIT] Starting Wikipedia extraction...")
    process_wiki_dump(XML_FILE)
