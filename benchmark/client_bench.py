import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import random
import argparse
import statistics
import uuid
import numpy as np
from contextlib import contextmanager
from Client.client import RedBoxClient

VECTORS    = 100_000
DIM        = 128
QUERIES    = 1_000
SEED       = 42

rng = np.random.default_rng(SEED)

def rand_vec() -> np.ndarray:
    return rng.random(DIM).astype(np.float32)

@contextmanager
def temp_db(host, port, capacity=None):
    cap    = capacity or VECTORS + 10_000    # headroom for warmup ops
    client = RedBoxClient(host=host, port=port, db_name=f"bench_{uuid.uuid4().hex[:10]}", dim=DIM, capacity=cap)
    try:
        yield client
    finally:
        client.drop()
        client.close()

def percentile(s, p):
    return s[int(len(s) * p)]

def print_stats(times_ms, count=None):
    s   = sorted(times_ms)
    n   = count or len(s)
    qps = n / (sum(times_ms) / 1000)
    print(f"   QPS  : {qps:,.1f} queries/sec")
    print(f"   Min  : {s[0]:.3f} ms")
    print(f"   Avg  : {statistics.mean(s):.3f} ms")
    print(f"   P50  : {percentile(s, 0.50):.3f} ms")
    print(f"   P95  : {percentile(s, 0.95):.3f} ms")
    print(f"   P99  : {percentile(s, 0.99):.3f} ms  <-- the one that matters")
    print(f"   Max  : {s[-1]:.3f} ms")

def sep(title):
    print(f"\n[{title}]")
    print("-" * 47)


def bench_insert(host, port):
    sep("1/6] INSERT THROUGHPUT")
    vecs = [rand_vec() for _ in range(VECTORS)]
    print(f"   Pre-generating {VECTORS:,} vectors...")

    with temp_db(host, port) as client:
        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)
        elapsed = time.perf_counter() - t0

    print(f"   Vectors    : {VECTORS:,}")
    print(f"   Time       : {elapsed:.3f} s")
    print(f"   Throughput : {VECTORS/elapsed:,.0f} vectors/sec")


def bench_search(host, port):
    sep("2/6] SEARCH LATENCY  (single nearest neighbor)")
    print("   Note: fresh DB, cold insert then queried immediately.")
    print("         Comparable to C++ hot-cache run.")
    print("-" * 47)

    vecs    = [rand_vec() for _ in range(VECTORS)]
    queries = [rand_vec() for _ in range(QUERIES)]

    with temp_db(host, port) as client:
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)

        for q in queries[:50]:
            client.search(q)

        times = []
        for q in queries:
            t0 = time.perf_counter()
            client.search(q)
            times.append((time.perf_counter() - t0) * 1000)

    print_stats(times)


def bench_search_n(host, port):
    sep("3/6] SEARCH_N LATENCY  (top-10 nearest neighbors)")
    K       = 10
    vecs    = [rand_vec() for _ in range(VECTORS)]
    queries = [rand_vec() for _ in range(QUERIES)]

    with temp_db(host, port) as client:
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)

        for q in queries[:50]:
            client.search_n(q, K)

        times = []
        for q in queries:
            t0 = time.perf_counter()
            client.search_n(q, K)
            times.append((time.perf_counter() - t0) * 1000)

    print(f"   K    : {K}")
    print_stats(times)


def bench_update(host, port):
    sep("4/6] UPDATE THROUGHPUT  (in-place via id_to_index)")
    vecs     = [rand_vec() for _ in range(VECTORS)]
    new_vecs = [rand_vec() for _ in range(QUERIES)]

    with temp_db(host, port) as client:
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)

        for i in range(50):
            client.update(i + 1, new_vecs[i % QUERIES])

        times = []
        for i in range(QUERIES):
            t0 = time.perf_counter()
            client.update(i + 1, new_vecs[i])
            times.append((time.perf_counter() - t0) * 1000)

    s   = sorted(times)
    qps = QUERIES / (sum(times) / 1000)
    print(f"   Updates    : {QUERIES:,}")
    print(f"   Throughput : {qps:,.0f} updates/sec")
    print(f"   Min  : {s[0]:.3f} ms")
    print(f"   Avg  : {statistics.mean(s):.3f} ms")
    print(f"   P50  : {percentile(s, 0.50):.3f} ms")
    print(f"   P95  : {percentile(s, 0.95):.3f} ms")
    print(f"   P99  : {percentile(s, 0.99):.3f} ms  <-- the one that matters")
    print(f"   Max  : {s[-1]:.3f} ms")
    print(f"   (O(1) lookup via id_to_index <-- no linear scan)")


def bench_mixed(host, port):
    sep("5/6] MIXED WORKLOAD  (70% search | 20% insert | 10% delete)")
    TOTAL_OPS = 10_000

    vecs = [rand_vec() for _ in range(VECTORS)]

    with temp_db(host, port) as client:
        ids = []
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)
            ids.append(i + 1)

        local_rng = random.Random(SEED)
        next_id   = VECTORS + 1
        searches  = inserts = deletes = 0
        t0        = time.perf_counter()

        for _ in range(TOTAL_OPS):
            r = local_rng.random()
            if r < 0.70:
                client.search(rand_vec())
                searches += 1
            elif r < 0.90:
                client.insert(next_id, rand_vec())
                ids.append(next_id)
                next_id += 1
                inserts += 1
            else:
                if ids:
                    client.delete(local_rng.choice(ids))
                    deletes += 1

        elapsed = time.perf_counter() - t0

    print(f"   Total Ops  : {TOTAL_OPS:,}")
    print(f"   Breakdown  : {searches} searches | {inserts} inserts | {deletes} deletes")
    print(f"   Total Time : {elapsed:.3f} s")
    print(f"   Throughput : {TOTAL_OPS/elapsed:,.0f} ops/sec")


def bench_search_under_deletion(host, port):
    sep("6/6] SEARCH UNDER HEAVY DELETION  (40% of DB deleted)")
    DELETE_COUNT = int(VECTORS * 0.4)
    vecs         = [rand_vec() for _ in range(VECTORS)]
    queries      = [rand_vec() for _ in range(QUERIES)]

    with temp_db(host, port) as client:
        for i, v in enumerate(vecs):
            client.insert(i + 1, v)

        for i in range(DELETE_COUNT):
            client.delete(i + 1)

        print(f"   Inserted   : {VECTORS:,} vectors")
        print(f"   Deleted    : {DELETE_COUNT:,} vectors (40%)")
        print(f"   Live rows  : {VECTORS - DELETE_COUNT:,}")
        print("-" * 47)

        for q in queries[:50]:
            client.search(q)

        times = []
        for q in queries:
            t0 = time.perf_counter()
            client.search(q)
            times.append((time.perf_counter() - t0) * 1000)

    print_stats(times)
    print("   (Compare QPS/P99 against Bench 2 to see deleted_flags impact)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--bench", default="all",
        help="Comma-separated bench IDs (1-6) or 'all'")
    args = parser.parse_args()

    benches = {
        "1": bench_insert,
        "2": bench_search,
        "3": bench_search_n,
        "4": bench_update,
        "5": bench_mixed,
        "6": bench_search_under_deletion,
    }

    selected = list(benches.keys()) if args.bench == "all" else [b.strip() for b in args.bench.split(",")]

    print("=" * 47)
    print("  RedBoxDb CLIENT BENCHMARK SUITE")
    print(f"  host={args.host}:{args.port}  dim={DIM}")
    print(f"  Vectors: {VECTORS:,}  Queries: {QUERIES:,}  Seed: {SEED}")
    print("=" * 47)
    print("  NOTE: All timings are end-to-end (TCP wire included)")

    t_total = time.perf_counter()
    for b in selected:
        if b in benches:
            benches[b](args.host, args.port)
        else:
            print(f"  [WARN] Unknown bench id '{b}', skipping")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 47}")
    print(f"  Total benchmark time: {elapsed:.1f}s")
    print("=" * 47)

if __name__ == "__main__":
    main()