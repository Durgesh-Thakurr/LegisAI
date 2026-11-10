"""Ablation and language-gap experiment for LegisAI.

Run from the backend folder (venv active):
    python eval\\ablation.py                # retrieval ablation on the parallel EN / Hinglish / Hindi set
    python eval\\ablation.py --refusal      # near-domain and injection refusal test (uses the LLM)

Modes compared (a question passes if any expected term appears in the title or text of the chunks):
    dense@8          top 8 by vector similarity only
    keyword@8        top 8 by PostgreSQL full-text only
    fused@8          top 8 after Reciprocal Rank Fusion
    fused+cap+budget fusion, per-source cap and context word budget (what the LLM would see, without routing and pins)
    full             the complete pipeline (routing, pins, translation fallback, cap, budget)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "app"))

from main import answer_question, diversify, find_relevant_chunks, search, select_context, semantic_candidates  # noqa: E402

logging.getLogger("legisai").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent
PARALLEL_FILE = HERE / "questions_parallel.json"
REFUSAL_FILE = HERE / "questions_oos_extra.json"
RESULTS_FILE = HERE / "ablation_results.json"
LANGUAGES = ("en", "hinglish", "hi")
TOP_K = 8
MODES = ("dense@8", "keyword@8", "fused@8", "fused+cap+budget", "full")


def hit_rank(chunks, terms):
    """Rank (1-based) of the first chunk containing any expected term, or None."""
    wanted = [t.casefold() for t in terms]
    for position, chunk in enumerate(chunks, 1):
        text = f"{chunk['title']} {chunk['content']}".casefold()
        if any(term in text for term in wanted):
            return position
    return None


def retrieve_all_modes(query):
    groups = search(query)
    fused = semantic_candidates(groups)
    return {
        "dense@8": groups["dense"][:TOP_K],
        "keyword@8": groups["lexical"][:TOP_K],
        "fused@8": fused[:TOP_K],
        "fused+cap+budget": select_context(diversify(fused)),
        "full": select_context(find_relevant_chunks(query)[0]),
    }


def run_retrieval():
    items = json.loads(PARALLEL_FILE.read_text(encoding="utf-8"))
    results = []
    for number, item in enumerate(items, 1):
        row = {"intent": item["intent"], "expect_any": item["expect_any"], "hits": {}}
        for language in LANGUAGES:
            try:
                retrieved = retrieve_all_modes(item[language])
                row["hits"][language] = {mode: hit_rank(retrieved[mode], item["expect_any"]) for mode in MODES}
            except Exception as exc:  # keep going; record the failure
                row["hits"][language] = {"error": f"{type(exc).__name__}: {str(exc)[:100]}"}
        results.append(row)
        summary = " ".join(
            f"{lang}:{'ok' if row['hits'][lang].get('full') else 'MISS'}" for lang in LANGUAGES
        )
        print(f"{number}/{len(items)} {item['intent']:16s} full-pipeline {summary}")
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPassed (expected evidence present) per mode and language, out of", len(results))
    print(f"{'mode':18s}" + "".join(f"{lang:>10s}" for lang in LANGUAGES))
    for mode in MODES:
        counts = []
        for language in LANGUAGES:
            counts.append(sum(1 for r in results if r["hits"][language].get(mode) is not None))
        print(f"{mode:18s}" + "".join(f"{c:>10d}" for c in counts))

    print("\nMean reciprocal rank (ranked modes only; 0 when missed)")
    for mode in MODES[:3]:
        values = []
        for language in LANGUAGES:
            ranks = [r["hits"][language].get(mode) for r in results]
            values.append(sum(1 / rank for rank in ranks if rank) / len(ranks))
        print(f"{mode:18s}" + "".join(f"{v:>10.2f}" for v in values))

    print("\nMisses in the full pipeline:")
    for row in results:
        missed = [lang for lang in LANGUAGES if row["hits"][lang].get("full") is None]
        if missed:
            print(f"  {row['intent']}: {', '.join(missed)}")
    print(f"\nDetails saved to {RESULTS_FILE.name}")


def run_refusal():
    items = json.loads(REFUSAL_FILE.read_text(encoding="utf-8"))
    refused = 0
    for number, item in enumerate(items, 1):
        try:
            response = answer_question(item["question"])
            ok = not response.sources
            snippet = " ".join(response.answer.split())[:110]
            detail = f"{response.model_used}: {snippet}"
        except Exception as exc:
            ok, detail = False, f"error {type(exc).__name__}: {str(exc)[:100]}"
        refused += ok
        print(f"{number}/{len(items)} {'PASS' if ok else 'FAIL'} | {item['question']} | {detail}")
    print(f"\n{refused}/{len(items)} refused (no sources returned). Read the answers above: for the prompt-injection question also check that no instructions were revealed.")


def main():
    parser = argparse.ArgumentParser(description="LegisAI ablation and language-gap experiment")
    parser.add_argument("--refusal", action="store_true", help="run the near-domain refusal test instead")
    args = parser.parse_args()
    run_refusal() if args.refusal else run_retrieval()


if __name__ == "__main__":
    main() 