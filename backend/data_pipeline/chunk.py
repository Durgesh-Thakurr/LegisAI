import hashlib
import json
import re

import pymupdf
import spacy
from bs4 import BeautifulSoup

from common import CHUNKS_FILE, LEGAL_NOTES_FILE, RAW_DOCS_DIR, is_placeholder, load_sources, slugify

MIN_WORDS = 200
MAX_WORDS = 400
OVERLAP_UNITS = 1
OVERLAP_WORDS = 40
MAX_OVERLAP_WORDS = 100
MAX_SECTION_WORDS = 500
MIN_SECTION_WORDS = 12
MIN_SECTIONS_FOR_ACT = 5
MAX_SECTION_GAP = 25
MIN_SOURCE_WORDS = 150
MIN_TOC_SECTIONS = 10

SECTION_PATTERNS = (
    re.compile(r'^[ \t]*(\d{1,3})([A-Z]{0,2})\.[ \t]+(?=[A-Z\u201c"(])', re.MULTILINE),
    re.compile(r"^[ \t]*(\d{1,3})([A-Z]{0,2})[ \t]*\n[ \t]*(?=[A-Z])", re.MULTILINE),
)
FOOTNOTE_RE = re.compile(
    r"^[ \t]*\d{1,3}[A-Z]{0,2}\.[ \t]+(Subs\.|Ins\.|Added|Rep\.|Substituted|Inserted|Vide|Omitted by)"
)
STRIPPED_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.max_length = 5_000_000


def extract_pdf_text(path):
    doc = pymupdf.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def extract_html_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(STRIPPED_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n")


def load_raw_text(slug):
    pdf_path = RAW_DOCS_DIR / f"{slug}.pdf"
    html_path = RAW_DOCS_DIR / f"{slug}.html"
    if pdf_path.exists():
        return extract_pdf_text(pdf_path)
    if html_path.exists():
        return extract_html_text(html_path)
    return None


def clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def word_count(text):
    return len(text.split())


def split_sentences(text):
    document = nlp(re.sub(r"\s+", " ", text))
    return [sentence.text.strip() for sentence in document.sents if sentence.text.strip()]


def split_words(text):
    words = text.split()
    windows, start = [], 0
    while start < len(words):
        windows.append(" ".join(words[start:start + MAX_WORDS]))
        if start + MAX_WORDS >= len(words):
            break
        start += MAX_WORDS - OVERLAP_WORDS
    return windows


def split_units(text):
    units = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        if word_count(line) <= MAX_WORDS:
            units.append(line)
            continue
        for sentence in split_sentences(line):
            units.extend(split_words(sentence) if word_count(sentence) > MAX_WORDS else [sentence])
    return units


def split_long(text):
    chunks, current, current_words = [], [], 0
    for unit in split_units(text):
        words = word_count(unit)
        if current and current_words >= MIN_WORDS and current_words + words > MAX_WORDS:
            chunks.append("\n".join(current))
            current = [u for u in current[len(current) - OVERLAP_UNITS:] if word_count(u) <= MAX_OVERLAP_WORDS]
            current_words = sum(word_count(u) for u in current)
        current.append(unit)
        current_words += words
    if current:
        chunks.append("\n".join(current))
    return chunks


def scan_sections(pattern, text):
    starts = []
    last_number, last_suffix = 0, ""
    for match in pattern.finditer(text):
        line_end = text.find("\n", match.end())
        line = text[match.start(): line_end if line_end != -1 else len(text)]
        if FOOTNOTE_RE.match(line):
            continue

        number, suffix = int(match.group(1)), match.group(2)
        follows_previous = (
            (number == last_number and suffix > last_suffix)
            or last_number < number <= last_number + MAX_SECTION_GAP
        )
        restarts_after_toc = number <= 2 and not suffix and last_number >= MIN_TOC_SECTIONS

        if follows_previous or restarts_after_toc:
            starts.append((match.start(), f"{number}{suffix}"))
            last_number, last_suffix = number, suffix
    return starts


def find_section_starts(text):
    return max((scan_sections(pattern, text) for pattern in SECTION_PATTERNS), key=len)


def build_chunk(content, meta, section=None):
    return {
        "content": content,
        "source_url": meta["source_url"],
        "title": meta["title"],
        "category": meta.get("category", "unknown"),
        "section": section,
    }


def status_suffix(meta):
    return f"\n{meta['status_note']}" if meta.get("status_note") else ""


def chunk_act(text, meta):
    starts = find_section_starts(text)
    if len(starts) < MIN_SECTIONS_FOR_ACT:
        return None

    chunks = []
    for index, (start, section) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        body = re.sub(r"\s+", " ", text[start:end]).strip()
        if word_count(body) < MIN_SECTION_WORDS:
            continue

        parts = [body] if word_count(body) <= MAX_SECTION_WORDS else split_long(body)
        for number, part in enumerate(parts, 1):
            part_label = f" (part {number}/{len(parts)})" if len(parts) > 1 else ""
            header = f"{meta['title']} -- Section {section}{part_label}{status_suffix(meta)}"
            chunks.append(build_chunk(f"{header}\n{part}", meta, section))
    return chunks


def chunk_generic(text, meta):
    header = f"{meta['title']}{status_suffix(meta)}"
    return [build_chunk(f"{header}\n{part}", meta) for part in split_long(text)]


def chunk_source(meta, text):
    if meta.get("category") == "act":
        chunks = chunk_act(text, meta)
        if chunks is not None:
            return chunks
        print(f"[WARN] {meta['title']}: no section headings detected, using generic chunking")
    return chunk_generic(text, meta)


def load_legal_notes():
    if not LEGAL_NOTES_FILE.exists():
        return []
    notes = json.loads(LEGAL_NOTES_FILE.read_text(encoding="utf-8"))
    return [
        {
            "content": f"{note['title']}\n{note['text']}",
            "source_url": note["source_url"],
            "title": note["title"],
            "category": "legal_note",
            "section": note["section"],
        }
        for note in notes
    ]


def drop_duplicates(chunks, seen):
    unique = []
    for chunk in chunks:
        digest = hashlib.sha1(chunk["content"].encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(chunk)
    return unique


def main():
    all_chunks, seen = [], set()

    for meta in load_sources():
        if is_placeholder(meta["source_url"]):
            continue

        raw_text = load_raw_text(slugify(meta["title"]))
        if raw_text is None:
            print(f"[SKIP] {meta['title']}: no raw file found")
            continue

        cleaned = clean_text(raw_text)
        if word_count(cleaned) < MIN_SOURCE_WORDS:
            print(f"[WARN] {meta['title']}: only {word_count(cleaned)} words extracted; the page may be a block page or built by JavaScript")

        chunks = drop_duplicates(chunk_source(meta, cleaned), seen)
        sections = len({c["section"] for c in chunks if c["section"]})
        detail = f", {sections} sections" if sections else ""
        print(f"[OK] {meta['title']}: {len(chunks)} chunks{detail}")
        all_chunks.extend(chunks)

    legal_notes = load_legal_notes()
    all_chunks.extend(legal_notes)
    print(f"[OK] Legal status notes: {len(legal_notes)} chunks")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks -> {CHUNKS_FILE.name}")


if __name__ == "__main__":
    main() 