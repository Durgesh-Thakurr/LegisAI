r"""Section-level golden tests for LegisAI retrieval. Run from the backend folder (venv active):

    python eval\golden_sections.py

Unlike run_eval.py (which only looks for an expected word anywhere in the context), every case here checks
three exact things about the chunks the LLM would receive:
  1. EXACT   the first chunk is the requested section of the requested Act
  2. ACT     when the question names an Act, the same section number from another Act is not in the context
  3. NOISE   at most one low-similarity filler chunk (< 0.5) rides along with an exact section hit

Exit code is non-zero if any case fails, so it can run on every code change.
"""
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "app"))

from main import chunk_section, find_relevant_chunks, select_context  # noqa: E402

logging.getLogger("legisai").setLevel(logging.WARNING)

IT, BNS, DPDP = "Information Technology Act", "Bharatiya Nyaya Sanhita", "Digital Personal Data Protection Act"
MAX_NOISE = 1
NOISE_BELOW = 0.5

# (question, Act the answer must come from, section, does the question name the Act?)
CASES = [
    ("What is section 66A?", IT, "66A", False),
    ("What does section 66E say?", IT, "66E", False),
    ("section 66F kya hai", IT, "66F", False),
    ("What is section 43A?", IT, "43A", False),
    ("What is section 67B of the IT Act?", IT, "67B", True),
    ("What is section 72A of the IT Act?", IT, "72A", True),
    ("What is section 43 of the IT Act?", IT, "43", True),
    ("section 66 of the IT Act", IT, "66", True),
    ("section 66 of BNS", BNS, "66", True),
    ("BNS dhara 318 kya hai?", BNS, "318", True),
    ("BNS section 308 extortion", BNS, "308", True),
    ("section 78 of the BNS", BNS, "78", True),
    ("BNS section 77 voyeurism", BNS, "77", True),
    ("DPDP Act section 8", DPDP, "8", True),
]


def check(question, act, section, names_act):
    context = select_context(find_relevant_chunks(question)[0])
    problems = []

    first = context[0] if context else None
    if not (first and first["title"].startswith(act) and chunk_section(first["content"]) == section):
        got = f"{first['title'][:30]} s.{chunk_section(first['content'])}" if first else "nothing"
        problems.append(f"first chunk is {got}, expected {act} s.{section}")

    if names_act:
        wrong = [c["title"][:26] for c in context if chunk_section(c["content"]) == section and not c["title"].startswith(act)]
        if wrong:
            problems.append(f"same section number from another Act in context: {wrong}")

    noise = sum(1 for c in context if c["similarity"] < 1.0 and c["similarity"] < NOISE_BELOW)
    if noise > MAX_NOISE:
        problems.append(f"{noise} low-similarity filler chunks (max {MAX_NOISE})")
    return problems, len(context), noise


def main():
    failed = 0
    for question, act, section, names_act in CASES:
        try:
            problems, size, noise = check(question, act, section, names_act)
        except Exception as exc:
            problems, size, noise = [f"error {type(exc).__name__}: {str(exc)[:100]}"], 0, 0
        status = "PASS" if not problems else "FAIL"
        failed += bool(problems)
        print(f"{status} | {question} | context {size} chunks, noise {noise}")
        for problem in problems:
            print(f"       - {problem}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()