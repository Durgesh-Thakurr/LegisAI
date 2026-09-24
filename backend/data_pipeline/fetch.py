import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

RAW_DOCS_DIR = Path("raw_docs")

APPROVED_DOMAINS = {
    "india.gov.in", "www.india.gov.in",
    "indiacode.nic.in", "www.indiacode.nic.in",
    "cybercrime.gov.in", "www.cybercrime.gov.in",
    "meity.gov.in", "www.meity.gov.in",
    "cert-in.org.in", "www.cert-in.org.in",
    "rbi.org.in", "www.rbi.org.in",
    "mha.gov.in", "www.mha.gov.in",
    "ncrb.gov.in", "www.ncrb.gov.in",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def slugify(title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80]


def load_sources(path="sources.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sources"]


def fetch_one(entry):
    url = entry["source_url"]
    title = entry["title"]

    if url.startswith("TODO"):
        print(f"[SKIP] {title}: placeholder URL")
        return

    domain = urlparse(url).netloc
    if domain not in APPROVED_DOMAINS:
        print(f"[REJECT] {title}: domain '{domain}' not approved -- refusing to fetch")
        return

    slug = slugify(title)
    is_pdf = url.lower().endswith(".pdf")
    out_path = RAW_DOCS_DIR / f"{slug}.{'pdf' if is_pdf else 'html'}"

    if out_path.exists():
        print(f"[SKIP] {title}: already downloaded ({out_path.name})")
        return

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[FAIL] {title}: {e}")
        return

    RAW_DOCS_DIR.mkdir(exist_ok=True)
    mode = "wb" if is_pdf else "w"
    content = resp.content if is_pdf else resp.text
    encoding_kwarg = {} if is_pdf else {"encoding": "utf-8"}

    with open(out_path, mode, **encoding_kwarg) as f:
        f.write(content)

    print(f"[OK] {title}: saved to {out_path}")


def main():
    sources = load_sources()
    print(f"Fetching {len(sources)} source(s) into {RAW_DOCS_DIR}/\n")
    for entry in tqdm(sources, desc="Fetching"):
        fetch_one(entry)


if __name__ == "__main__":
    main()