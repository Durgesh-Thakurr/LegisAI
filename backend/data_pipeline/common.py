import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

BASE_DIR = Path(__file__).resolve().parent
RAW_DOCS_DIR = BASE_DIR / "raw_docs"
SOURCES_FILE = BASE_DIR / "sources.json"
CHUNKS_FILE = BASE_DIR / "chunks.json"

APPROVED_DOMAINS = {
    "india.gov.in", "www.india.gov.in",
    "indiacode.nic.in", "www.indiacode.nic.in",
    "cybercrime.gov.in", "www.cybercrime.gov.in",
    "meity.gov.in", "www.meity.gov.in",
    "cert-in.org.in", "www.cert-in.org.in",
    "rbi.org.in", "www.rbi.org.in",
    "mha.gov.in", "www.mha.gov.in",
    "ncrb.gov.in", "www.ncrb.gov.in",
    "services.india.gov.in",
    "cag.gov.in", "www.cag.gov.in",
    "police.py.gov.in",
    "i4c.mha.gov.in",
    "cyber.delhipolice.gov.in",
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
MAX_REDIRECTS = 5


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def load_sources(path=SOURCES_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["sources"]


def is_placeholder(url):
    return url.startswith("TODO")


def is_approved_url(url):
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in APPROVED_DOMAINS


def safe_get(url, timeout=(10, 30)):
    """GET that re-checks the approved-domain list on every redirect hop.

    Returns an open streaming response that the caller must close.
    """
    for _ in range(MAX_REDIRECTS + 1):
        if not is_approved_url(url):
            raise ValueError(f"URL not approved: {urlparse(url).hostname}")
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=timeout, stream=True, allow_redirects=False
        )
        if not response.is_redirect:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise ValueError("redirect without Location header")
        url = urljoin(url, location)
    raise ValueError("too many redirects") 