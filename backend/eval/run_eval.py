import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "app"))

from main import find_relevant_chunks  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.json")
RESULTS_FILE = Path(__file__).with_name("last_run.log")

logging.getLogger("legisai").setLevel(logging.WARNING)


def evaluate(item):
    chunks, _ = find_relevant_chunks(item["question"])
    haystack = " ".join(f"{c['title']} {c['content']}" for c in chunks).casefold()
    matched = [term for term in item["expect_any"] if term.casefold() in haystack]
    titles = sorted({c["title"][:40] for c in chunks})
    return bool(matched), titles


def record(line):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tally():
    lines = RESULTS_FILE.read_text(encoding="utf-8").splitlines()
    results = [line for line in lines if line.startswith(("PASS", "FAIL"))]
    return sum(line.startswith("PASS") for line in results), len(results)


def main():
    parser = argparse.ArgumentParser(description="Measure whether retrieval surfaces the expected facts")
    parser.add_argument("--start", type=int, default=1, help="resume from this question number")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit non-zero below this pass rate (0-1)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.start == 1:
        RESULTS_FILE.write_text("", encoding="utf-8")

    for number, item in enumerate(questions, 1):
        if number < args.start:
            continue
        ok, titles = evaluate(item)
        if ok:
            line = f"PASS | {item['question']}"
        else:
            line = f"FAIL | {item['question']} | expected any of {item['expect_any']} | retrieved from {titles}"
        record(line)
        print(f"{number}/{len(questions)} {line}")

    passed, total = tally()
    print(f"\n{passed}/{total} passed ({passed / total:.0%}). Details: {RESULTS_FILE.name}")
    if total < len(questions):
        print(f"Only {total} of {len(questions)} questions recorded.")
    if passed / total < args.min_pass:
        sys.exit(1)


if __name__ == "__main__":
    main() 