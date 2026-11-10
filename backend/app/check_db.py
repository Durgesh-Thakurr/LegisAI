import os
import time
import warnings

import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

REPEATS = 5
SEARCH_SQL = "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 24"


def seconds(fn):
    started = time.monotonic()
    result = fn()
    return time.monotonic() - started, result


def plan_nodes(node):
    yield node["Node Type"]
    for child in node.get("Plans", []):
        yield from plan_nodes(child)


def main():
    elapsed, conn = seconds(lambda: psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=15))
    conn.autocommit = True
    cur = conn.cursor()
    print(f"new connection            : {elapsed:.2f}s")

    def run(sql, params=None):
        cur.execute(sql, params)
        return cur.fetchall()

    round_trips = [seconds(lambda: run("SELECT 1"))[0] for _ in range(REPEATS)]
    print("round trip (SELECT 1)     : " + "  ".join(f"{t:.2f}s" for t in round_trips))

    vector = run("SELECT embedding FROM chunks LIMIT 1")[0][0]
    from_here = [seconds(lambda: run(SEARCH_SQL, (vector,)))[0] for _ in range(REPEATS)]
    print("vector search, seen here  : " + "  ".join(f"{t:.2f}s" for t in from_here))

    on_server, nodes = [], []
    for _ in range(REPEATS):
        plan = run("EXPLAIN (ANALYZE, FORMAT JSON) " + SEARCH_SQL, (vector,))[0][0][0]
        on_server.append(plan["Execution Time"])
        nodes = list(plan_nodes(plan["Plan"]))
    print("vector search, on server  : " + "  ".join(f"{t:.0f}ms" for t in on_server))
    print("query plan                : " + " > ".join(nodes))

    median = sorted(round_trips)[REPEATS // 2]
    server_median = sorted(on_server)[REPEATS // 2] / 1000
    print()
    if median > 0.25:
        print(f"Verdict: each round trip costs ~{median:.2f}s. The network to the database is the slow part.")
        print("         Hosting the backend in the same region as the database removes most of it.")
    elif server_median > 0.25:
        print("Verdict: the database itself is slow on this query (cold storage or a small compute).")
    else:
        print("Verdict: the database is fast right now. Earlier slowness was a cold start or a busy network.")


if __name__ == "__main__":
    main()