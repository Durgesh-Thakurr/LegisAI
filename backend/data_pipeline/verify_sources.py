import sys

import requests

from common import is_approved_url, is_placeholder, load_sources, safe_get

STATUSES = ("OK", "SKIP", "REJECT", "FAIL")


def check_source(entry):
    url, title = entry["source_url"], entry["title"]

    if is_placeholder(url):
        return "SKIP", f"{title}: placeholder URL"
    if not is_approved_url(url):
        return "REJECT", f"{title}: not https on an approved domain"

    try:
        response = safe_get(url, timeout=(10, 15))
    except (requests.RequestException, ValueError) as exc:
        return "FAIL", f"{title}: {exc}"

    try:
        if response.status_code >= 400:
            return "FAIL", f"{title}: HTTP {response.status_code}"
        return "OK", f"{title}: reachable ({response.status_code})"
    finally:
        response.close()


def main():
    counts = dict.fromkeys(STATUSES, 0)
    for entry in load_sources():
        status, message = check_source(entry)
        counts[status] += 1
        print(f"[{status}] {message}")

    print("\n" + "  ".join(f"{status}: {counts[status]}" for status in STATUSES))
    if counts["REJECT"] or counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main() 