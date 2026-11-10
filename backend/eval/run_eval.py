import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "app"))

from main import CONTEXT_CHUNKS, CONTEXT_WORD_BUDGET, answer_question, find_relevant_chunks, select_context  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.json")

logging.getLogger("legisai").setLevel(logging.WARNING)


def results_file(answers_mode):
    return Path(__file__).with_name("last_answers.log" if answers_mode else "last_run.log")


def check_retrieval(item):
    chunks, top_similarity = find_relevant_chunks(item["question"])
    chunks = select_context(chunks)
    titles = sorted({c["title"][:40] for c in chunks})
    haystack = " ".join(f"{c['title']} {c['content']}" for c in chunks).casefold()
    matched = [term for term in item["expect_any"] if term.casefold() in haystack]
    detail = f"expected any of {item['expect_any']} | retrieved from {titles} | top {top_similarity:.2f}"
    return bool(matched), detail


def check_refusal(item):
    response = answer_question(item["question"])
    answer = " ".join(response.answer.split())[:90]
    return not response.sources, f"expected a refusal | answered by {response.model_used}: {answer}"


def record(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tally(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    results = [line for line in lines if line.startswith(("PASS", "FAIL"))]
    return sum(line.startswith("PASS") for line in results), len(results)


def main():
    parser = argparse.ArgumentParser(description="Measure retrieval, or with --answers whether out-of-scope questions are refused")
    parser.add_argument("--answers", action="store_true", help="run the out-of-scope questions through the full pipeline (uses the LLM)")
    parser.add_argument("--start", type=int, default=1, help="resume from this question number")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit non-zero below this pass rate (0-1)")
    args = parser.parse_args()

    path = results_file(args.answers)
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.start == 1:
        path.write_text("", encoding="utf-8")

    if args.answers:
        print("Checking that out-of-scope questions are refused by the full pipeline.\n")
        check = check_refusal
    else:
        print(f"Checking the chunks the LLM receives (up to {CONTEXT_CHUNKS} chunks or {CONTEXT_WORD_BUDGET} words).\n")
        check = check_retrieval

    for number, item in enumerate(questions, 1):
        if number < args.start or item.get("expect_none", False) != args.answers:
            continue
        try:
            ok, detail = check(item)
        except Exception as exc:
            ok, detail = False, f"error {type(exc).__name__}: {str(exc)[:120]}"
        line = f"PASS | {item['question']}" if ok else f"FAIL | {item['question']} | {detail}"
        record(path, line)
        print(f"{number}/{len(questions)} {line}")

    passed, total = tally(path)
    print(f"\n{passed}/{total} passed ({passed / total:.0%}). Details: {path.name}")
    if passed / total < args.min_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()