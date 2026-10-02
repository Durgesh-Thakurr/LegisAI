import argparse
import sys

import requests
from tqdm import tqdm

from common import RAW_DOCS_DIR, is_approved_url, is_placeholder, load_sources, safe_get, slugify

MAX_BYTES = 50 * 1024 * 1024
PDF_MAGIC = b"%PDF-"


def download(url):
    response = safe_get(url)
    try:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        parts, size = [], 0
        for part in response.iter_content(65536):
            size += len(part)
            if size > MAX_BYTES:
                raise ValueError("file exceeds 50 MB limit")
            parts.append(part)
        return b"".join(parts), content_type
    finally:
        response.close()


def fetch_one(entry, force):
    url, title = entry["source_url"], entry["title"]

    if is_placeholder(url):
        print(f"[SKIP] {title}: placeholder URL")
        return "skip"
    if not is_approved_url(url):
        print(f"[FAIL] {title}: not https on an approved domain")
        return "fail"

    slug = slugify(title)
    pdf_path = RAW_DOCS_DIR / f"{slug}.pdf"
    html_path = RAW_DOCS_DIR / f"{slug}.html"

    if not force and (pdf_path.exists() or html_path.exists()):
        print(f"[SKIP] {title}: already downloaded")
        return "skip"

    try:
        data, content_type = download(url)
    except (requests.RequestException, ValueError) as exc:
        print(f"[FAIL] {title}: {exc}")
        return "fail"

    is_pdf = url.lower().endswith(".pdf") or "pdf" in content_type or data.startswith(PDF_MAGIC)
    if is_pdf and not data.startswith(PDF_MAGIC):
        print(f"[FAIL] {title}: expected a PDF but received something else")
        return "fail"

    out_path = pdf_path if is_pdf else html_path
    RAW_DOCS_DIR.mkdir(exist_ok=True)
    temp_path = out_path.with_name(out_path.name + ".part")
    temp_path.write_bytes(data)
    temp_path.replace(out_path)

    print(f"[OK] {title}: {out_path.name} ({len(data) // 1024} KB)")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Download sources.json entries into raw_docs/")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    force = parser.parse_args().force

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for entry in tqdm(load_sources(), desc="Fetching"):
        counts[fetch_one(entry, force)] += 1

    print(f"\nOK: {counts['ok']}  SKIP: {counts['skip']}  FAIL: {counts['fail']}")
    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main() 