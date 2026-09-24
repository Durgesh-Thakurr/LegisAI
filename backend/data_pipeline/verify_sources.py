"""
Verify every entry in sources.json before ingesting anything.
Run this every time sources.json changes.
"""

import json
import sys
from urllib.parse import urlparse

import requests

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


def load_sources(path="sources.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sources"]


def check_source(entry):
    url = entry["source_url"]
    title = entry["title"]

    if url.startswith("TODO"):
        return "SKIP", f"{title}: placeholder URL not filled in yet"

    domain = urlparse(url).netloc
    if domain not in APPROVED_DOMAINS:
        return "REJECT", f"{title}: domain '{domain}' not on approved list"

    try:
        resp = requests.head(url, timeout=15, allow_redirects=True, headers=HEADERS)
        if resp.status_code >= 400:
            resp = requests.get(url, timeout=15, stream=True, headers=HEADERS)
        if resp.status_code >= 400:
            return "FAIL", f"{title}: HTTP {resp.status_code}"
        return "OK", f"{title}: reachable ({resp.status_code})"
    except requests.RequestException as e:
        return "FAIL", f"{title}: request error -- {e}"


def main():
    sources = load_sources()
    results = {"OK": [], "SKIP": [], "REJECT": [], "FAIL": []}

    for entry in sources:
        status, message = check_source(entry)
        results[status].append(message)
        print(f"[{status}] {message}")

    print("\n--- Summary ---")
    for status in ("OK", "SKIP", "REJECT", "FAIL"):
        print(f"{status}: {len(results[status])}")

    if results["REJECT"]:
        print("\nFix REJECT entries before running the ingestion pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main() 