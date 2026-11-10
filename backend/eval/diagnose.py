import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from main import diversify, search, select_context, semantic_candidates  # noqa: E402

QUERY = "cyber stalking under Bharatiya Nyaya Sanhita"
TARGET = "-- Section 78"


def rank_of(chunks):
    for position, chunk in enumerate(chunks, 1):
        if "Nyaya" in chunk["title"] and TARGET in chunk["content"][:200]:
            return position
    return None


groups = search(QUERY)
fused = semantic_candidates(groups)
diversified = diversify(fused)

print(f"\ndense   : rank {rank_of(groups['dense'])} of {len(groups['dense'])}")
print(f"lexical : rank {rank_of(groups['lexical'])} of {len(groups['lexical'])}")
print(f"fused   : rank {rank_of(fused)} of {len(fused)}")
print(f"after per-source cap (top 8): rank {rank_of(diversified)}")
print(f"in final LLM context        : rank {rank_of(select_context(diversified))}")

print("\nfused top 12:")
for position, chunk in enumerate(fused[:12], 1):
    flag = "L" if chunk["lexical"] else "-"
    print(f"{position:2d} {chunk['similarity']:.2f} {flag} {chunk['title'][:28]} | {chunk['content'][:70]!r}") 