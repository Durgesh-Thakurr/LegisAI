import json
import re
from pathlib import Path

import pymupdf
import spacy
from bs4 import BeautifulSoup

RAW_DOCS_DIR = Path("raw_docs")
OUTPUT_FILE = Path("chunks.json")
CHUNK_MIN_TOKENS = 200
CHUNK_MAX_TOKENS = 400

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def load_sources(path="sources.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["sources"]


def extract_pdf_text(path):
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_html_text(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, source_meta):
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    current, current_tokens = [], 0

    for sent in sentences:
        sent_tokens = len(sent.split())
        if current_tokens + sent_tokens > CHUNK_MAX_TOKENS and current_tokens >= CHUNK_MIN_TOKENS:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return [
        {
            "content": chunk,
            "source_url": source_meta["source_url"],
            "title": source_meta["title"],
            "category": source_meta.get("category", "unknown"),
        }
        for chunk in chunks
    ]


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def main():
    sources = load_sources()
    all_chunks = []

    for entry in sources:
        if entry["source_url"].startswith("TODO"):
            continue

        slug = slugify(entry["title"])
        pdf_path = RAW_DOCS_DIR / f"{slug}.pdf"
        html_path = RAW_DOCS_DIR / f"{slug}.html"

        if pdf_path.exists():
            raw_text = extract_pdf_text(pdf_path)
        elif html_path.exists():
            raw_text = extract_html_text(html_path)
        else:
            print(f"[SKIP] {entry['title']}: no raw file found")
            continue

        text = clean_text(raw_text)
        chunks = chunk_text(text, entry)
        all_chunks.extend(chunks)
        print(f"[OK] {entry['title']}: {len(chunks)} chunks")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)} -> saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 