"""Sanity checks for chunks.json. Run after chunk.py and before embed_and_store.py:

    python verify_chunks.py [path/to/chunks.json]

Exits non-zero if any check fails, so a broken chunker can never reach the database.
"""
import json
import re
import sys

from common import CHUNKS_FILE

EXPECTED_SECTIONS = {
    "Information Technology Act": [
        "2", "43", "43A", "65", "66", "66A", "66B", "66C", "66D", "66E", "66F", "67", "67A", "67B", "67C",
        "69", "69A", "69B", "70", "72", "72A", "79", "84A", "84B", "84C",
    ],
    "Digital Personal Data Protection Act": [str(n) for n in range(1, 45)],
    "Bharatiya Nyaya Sanhita": [str(n) for n in range(1, 359)],
}
STATUS_CHECKS = (
    ("Information Technology Act", "66A", "struck down"),
    ("Information Technology Act", "43A", "44(2)"),
)
HEADER_PART = re.compile(r"\(part (\d+)/\d+\)")


def body_of(chunk):
    lines = chunk["content"].split("\n")
    start = 1
    while start < len(lines) and lines[start].startswith("LEGAL STATUS NOTE"):
        start += 1
    return " ".join(lines[start:]).strip()


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    failures = []

    for prefix, expected in EXPECTED_SECTIONS.items():
        own = [c for c in chunks if c["title"].startswith(prefix) and c["category"] == "act"]
        present = {c["section"] for c in own if c["section"]}
        missing = [s for s in expected if s not in present]
        if missing:
            failures.append(f"{prefix}: {len(missing)} expected sections missing: {missing[:20]}")

        mismatched = []
        for chunk in own:
            first_line = chunk["content"].split("\n")[0]
            part = HEADER_PART.search(first_line)
            if part and part.group(1) != "1":
                continue
            found = re.match(r"\s*(\d{1,3})\s?([A-Z]{0,2})\b", body_of(chunk))
            if not (found and found.group(1) + found.group(2) == chunk["section"]):
                mismatched.append(chunk["section"])
        if mismatched:
            failures.append(f"{prefix}: {len(mismatched)} chunks whose text does not start with their section label: {mismatched[:15]}")
        print(f"[{'FAIL' if missing or mismatched else 'OK'}] {prefix}: {len(own)} chunks, {len(present)} sections")

    for prefix, section, needle in STATUS_CHECKS:
        found = [c for c in chunks if c["title"].startswith(prefix) and c["section"] == section and c["category"] == "act"]
        if not found or not all(needle in c["content"] for c in found):
            failures.append(f"{prefix} section {section}: status note containing '{needle}' is missing")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print(f"\nAll checks passed on {len(chunks)} chunks.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CHUNKS_FILE) 