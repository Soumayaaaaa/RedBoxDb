import uuid
import argparse
import numpy as np
from contextlib import contextmanager
from Client.client import RedBoxClient

DIM     = 128
QUERIES = 200

def make_corpus(db_size, seed) -> np.ndarray:
    return np.random.default_rng(seed).random((db_size, DIM)).astype(np.float32)

def make_queries(seed) -> np.ndarray:
    return np.random.default_rng(seed + 999).random((QUERIES, DIM)).astype(np.float32)

def brute_force_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    results = np.empty((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        dists = np.sum((corpus - q) ** 2, axis=1)
        results[i] = np.argsort(dists)[:k]
    return results

@contextmanager
def temp_db(host, port, capacity):
    client = RedBoxClient(
        host=host, port=port,
        db_name=f"recall_{uuid.uuid4().hex[:10]}",
        dim=DIM, capacity=capacity
    )
    try:
        yield client
    finally:
        try:
            client.drop()
        except Exception:
            pass
        client.close()

def sanity_check(host, port):
    print("  Running sanity check...")
    vec = np.ones(DIM, dtype=np.float32)
    with temp_db(host, port, capacity=100) as client:
        client.insert(1, vec)
        result = client.search(vec)
        if result != 1:
            print(f"  [FAIL] Inserted ID=1, search returned ID={result}")
            print("         Stale .db files on disk — delete all *.db and *.db.del files and restart server.")
            return False
        print(f"  [PASS] Inserted ID=1, search returned ID={result}")
        return True

def run_recall(host, port, db_size, k, seed=42) -> float:
    corpus  = make_corpus(db_size, seed)
    queries = make_queries(seed)
    true_topk = brute_force_topk(corpus, queries, k)

    with temp_db(host, port, capacity=db_size + 1000) as client:
        for i, vec in enumerate(corpus):
            client.insert(i + 1, vec)

        hits = 0
        for i, q in enumerate(queries):
            ivf_ids  = set(client.search_n(q, k))
            true_ids = set(int(x) + 1 for x in true_topk[i])
            hits += len(ivf_ids & true_ids)

    return hits / (QUERIES * k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    DB_SIZES = [9_999, 50_000, 100_000]
    K_VALUES = [1]

    print("=" * 52)
    print("  RedBoxDb RECALL TEST  (IVF vs Brute Force L2)")
    print(f"  dim={DIM}  queries={QUERIES}")
    print("=" * 52)

    if not sanity_check(args.host, args.port):
        return

    print()
    print(f"  {'DB Size':<12} {'K':<6} {'Recall@K':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*10}")

    for db_size in DB_SIZES:
        for k in K_VALUES:
            recall = run_recall(args.host, args.port, db_size, k)
            flag = "  <-- BAD" if recall < 0.8 else ""
            print(f"  {db_size:<12,} {k:<6} {recall:>9.1%}{flag}")
        print()

if __name__ == "__main__":
    main()