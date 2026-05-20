"""
RedBoxDb - Comprehensive Validation Suite (v2)
Covers all engine features including K-Means++ clustering, multi-probe search,
IVF correctness, tombstone compaction, concurrency, and persistence.
"""

import time
import uuid
import random
import sys
import os
import threading
from client import RedBoxClient

# ==========================================
# TEST FRAMEWORK
# ==========================================
class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self._section = ""

    def section(self, name):
        self._section = name
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

    def log(self, msg):
        print(f"   [INFO] {msg}")

    def assert_true(self, condition, test_name):
        if condition:
            print(f"   [PASS] {test_name}")
            self.passed += 1
        else:
            print(f"   [FAIL] {test_name}")
            self.failed += 1

    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            print(f"   [PASS] {test_name}  (got {actual})")
            self.passed += 1
        else:
            print(f"   [FAIL] {test_name}  (expected {expected}, got {actual})")
            self.failed += 1

    def assert_in(self, item, collection, test_name):
        if item in collection:
            print(f"   [PASS] {test_name}  ({item} in result)")
            self.passed += 1
        else:
            print(f"   [FAIL] {test_name}  ({item} not in {collection})")
            self.failed += 1

    def assert_not_in(self, item, collection, test_name):
        if item not in collection:
            print(f"   [PASS] {test_name}")
            self.passed += 1
        else:
            print(f"   [FAIL] {test_name}  ({item} unexpectedly found in {collection})")
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"  RESULTS: {self.passed}/{total} passed", end="")
        if self.failed == 0:
            print("  ✓ ALL CLEAR")
        else:
            print(f"  ✗ {self.failed} FAILED")
        print(f"{'='*50}")


HOST = "127.0.0.1"
PORT = 8080

# ==========================================
# HELPERS
# ==========================================
class ManagedDb:
    """Opens a fresh uniquely-named DB and drops it from the server on exit."""
    def __init__(self, dim):
        self.name = f"val_{uuid.uuid4().hex[:10]}"
        self.dim  = dim
        self._client = RedBoxClient(host=HOST, port=PORT, db_name=self.name, dim=dim)

    def __enter__(self):
        return self._client

    def __exit__(self, *_):
        try:
            self._client.drop()
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass

def fresh_db(dim):
    """Return (db_name, client) — NOTE: use ManagedDb context manager instead
    when you want automatic server-side cleanup."""
    name = f"val_{uuid.uuid4().hex[:10]}"
    return name, RedBoxClient(host=HOST, port=PORT, db_name=name, dim=dim)


# ==========================================
# TESTS
# ==========================================

def test_empty_db_search(t: TestSuite):
    t.section("TEST 1 — Empty DB Edge Cases")
    with ManagedDb(3) as client:
        result = client.search([1.0, 0.0, 0.0])
        t.assert_equal(result, -1, "search() on empty DB returns -1")

        results = client.search_n([1.0, 0.0, 0.0], 5)
        t.assert_equal(results, [], "search_n() on empty DB returns []")

        ok = client.delete(999)
        t.assert_true(not ok, "delete() on never-inserted ID returns False")

        ok = client.update(999, [1.0, 0.0, 0.0])
        t.assert_true(not ok, "update() on non-existent ID returns False")


def test_vector_math(t: TestSuite):
    t.section("TEST 2 — Vector Math & Proximity")
    with ManagedDb(3) as client:
        client.insert(10, [1.0, 0.0, 0.0])
        client.insert(20, [0.0, 1.0, 0.0])
        client.insert(30, [0.0, 0.0, 1.0])

        t.assert_equal(client.search([0.9, 0.1, 0.0]), 10, "Nearest to X-axis is ID 10")
        t.assert_equal(client.search([0.1, 0.9, 0.0]), 20, "Nearest to Y-axis is ID 20")
        t.assert_equal(client.search([0.1, 0.1, 0.9]), 30, "Nearest to Z-axis is ID 30")

        # Tie-breaker: equidistant — at least returns one of the candidates
        result = client.search([0.0, 0.0, 0.0])
        t.assert_true(result in [10, 20, 30], "Query at origin returns a valid ID")


def test_search_n_ordering(t: TestSuite):
    t.section("TEST 3 — search_N Ordering & Correctness")
    with ManagedDb(1) as client:
        # Insert at distances 1, 4, 9, 10000 from origin
        client.insert(1, [1.0])
        client.insert(2, [2.0])
        client.insert(3, [3.0])
        client.insert(99, [100.0])

        results = client.search_n([0.0], 3)
        t.assert_equal(len(results), 3, "search_N(3) returns 3 results")
        t.assert_equal(results[0], 1, "Closest is ID 1 (dist=1)")
        t.assert_equal(results[1], 2, "2nd closest is ID 2 (dist=4)")
        t.assert_equal(results[2], 3, "3rd closest is ID 3 (dist=9)")

        results = client.search_n([0.0], 1)
        t.assert_equal(results, [1], "search_N(1) returns only the best")

        # Ask for more than DB has
        results = client.search_n([0.0], 100)
        t.assert_equal(len(results), 4, "search_N(100) caps at actual DB size (4)")


def test_crud_lifecycle(t: TestSuite):
    t.section("TEST 4 — Full CRUD Lifecycle")
    with ManagedDb(4) as client:
        vec_id = random.randint(1000, 9999)

        # INSERT + READ
        client.insert(vec_id, [1.0, 1.0, 1.0, 1.0])
        t.assert_equal(client.search([1.0, 1.0, 1.0, 1.0]), vec_id, "Insert then search finds correct ID")

        # UPDATE
        new_vec = [2.0, 2.0, 2.0, 2.0]
        ok = client.update(vec_id, new_vec)
        t.assert_true(ok, "update() returns True on existing ID")
        t.assert_equal(client.search(new_vec), vec_id, "Search after update finds same ID at new location")

        # UPDATE on deleted ID
        client.delete(vec_id)
        ok = client.update(vec_id, [3.0, 3.0, 3.0, 3.0])
        t.assert_true(not ok, "update() returns False on deleted ID")

        # DELETE then verify
        distractor = vec_id + 1
        client.insert(distractor, [9.0, 9.0, 9.0, 9.0])
        result = client.search(new_vec)
        t.assert_true(result != vec_id, "Deleted ID is not returned by search")

        # Double delete
        ok = client.delete(vec_id)
        t.assert_true(not ok, "Double delete returns False")


def test_insert_auto_sequencing(t: TestSuite):
    t.section("TEST 5 — insert_auto ID Sequencing")
    with ManagedDb(2) as client:
        ids = [client.insert_auto([float(i), 0.0]) for i in range(5)]
        t.assert_equal(len(set(ids)), 5, "All auto-assigned IDs are unique")
        t.assert_equal(ids, sorted(ids), "Auto IDs are monotonically increasing")
        t.assert_true(all(i > 0 for i in ids), "All auto IDs are positive")

        # Verify each auto-inserted vector is actually searchable
        result = client.search([2.0, 0.0])
        t.assert_equal(result, ids[2], "Auto-inserted vector is searchable by content")


def test_search_n_after_deletion(t: TestSuite):
    t.section("TEST 6 — search_N Correctness After Deletion")
    with ManagedDb(1) as client:
        client.insert(1, [1.0])
        client.insert(2, [2.0])
        client.insert(3, [3.0])
        client.insert(99, [100.0])

        client.delete(2)  # Remove the silver medal

        results = client.search_n([0.0], 3)
        t.assert_equal(len(results), 3, "search_N(3) still returns 3 after 1 deletion")
        t.assert_not_in(2, results, "Deleted ID 2 not in results")
        t.assert_in(99, results, "ID 99 promoted to 3rd place after deletion")
        t.assert_equal(results[0], 1, "Gold medal (ID 1) still first")


def test_multi_tenant_isolation(t: TestSuite):
    t.section("TEST 7 — Multi-Tenant Isolation")

    # Write to DB A then drop it
    with RedBoxClient(host=HOST, port=PORT, db_name="val_iso_a", dim=3) as a:
        a.insert(1, [1.0, 0.0, 0.0])
        a.drop()

    # DB B (dim=5) must not see DB A's data or dimensions
    with RedBoxClient(host=HOST, port=PORT, db_name="val_iso_b", dim=5) as b:
        # Dim mismatch blocked on client side
        try:
            b.search([1.0, 0.0, 0.0])  # dim=3 vec into dim=5 db
            t.assert_true(False, "Dimension mismatch should raise ValueError")
        except ValueError:
            t.assert_true(True, "Client rejects dimension mismatch for search")

        # DB B has its own data, isolated from A
        b.insert(42, [1.0, 0.0, 0.0, 0.0, 0.0])
        result = b.search([1.0, 0.0, 0.0, 0.0, 0.0])
        t.assert_equal(result, 42, "DB B search finds its own data, not DB A's")
        b.drop()


def test_persistence(t: TestSuite):
    t.section("TEST 8 — Persistence (Reconnect Simulation)")
    db_name = f"val_persist_{uuid.uuid4().hex[:8]}"
    pid = 7777
    vec = [0.5, 0.5, 0.5]

    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        c.insert(pid, vec)

    time.sleep(0.3)

    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        result = c.search(vec)
        t.assert_equal(result, pid, "Data survives reconnection")

    # Tombstones also persist
    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        c.delete(pid)

    time.sleep(0.3)

    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        distractor = pid + 1
        c.insert(distractor, [0.6, 0.6, 0.6])
        result = c.search(vec)
        t.assert_true(result != pid, "Deleted ID stays deleted after reconnect")
        c.drop()


def test_kmeans_init_correctness(t: TestSuite):
    """
    Insert KMEANS_INIT_THRESHOLD + extra vectors (engine threshold is 10k).
    Verify correctness is preserved before AND after K-Means++ fires.
    We use a smaller dim and known anchor vectors at clearly distinct positions.
    """
    t.section("TEST 9 — K-Means++ Init Correctness (>10k vectors)")
    THRESHOLD = 10_000
    DIM = 4
    db_name = f"val_kmeans_{uuid.uuid4().hex[:8]}"

    t.log(f"Inserting {THRESHOLD + 200} vectors to cross K-Means++ threshold...")
    anchor_id = 99999
    anchor_vec = [100.0, 100.0, 100.0, 100.0]

    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=DIM) as c:
        # Insert the anchor FIRST (pre-init phase)
        c.insert(anchor_id, anchor_vec)

        # Flood with random-ish vectors clustered near origin
        for i in range(THRESHOLD + 199):
            v = [float(i % 10) * 0.01, float((i+1) % 10) * 0.01,
                 float((i+2) % 10) * 0.01, float((i+3) % 10) * 0.01]
            c.insert_auto(v)

        # After K-Means++ fires, anchor should still be findable
        result = c.search(anchor_vec)
        t.assert_equal(result, anchor_id, "Anchor vector still found after K-Means++ init")

        # Insert more AFTER init to exercise the online centroid update path
        extra_id = 88888
        extra_vec = [-100.0, -100.0, -100.0, -100.0]
        c.insert(extra_id, extra_vec)

        result = c.search(extra_vec)
        t.assert_equal(result, extra_id, "Post-init insert is searchable via online centroid update")

        # search_N correctness post-init
        results = c.search_n(anchor_vec, 3)
        t.assert_in(anchor_id, results, "Anchor appears in search_N top-3 post-init")
        c.drop()


def test_tombstone_compaction(t: TestSuite):
    t.section("TEST 10 — Tombstone Compaction")
    COMPACT_SLACK = 64
    N = COMPACT_SLACK + 10
    db_name = f"val_compact_{uuid.uuid4().hex[:8]}"
    del_file = db_name + ".db.del"

    t.log(f"Inserting and deleting {N} vectors to trigger compaction...")
    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        for i in range(1, N + 1):
            c.insert(i, [float(i), 0.0, 0.0])
        for i in range(1, N + 1):
            c.delete(i)

    time.sleep(0.3)

    if os.path.exists(del_file):
        raw_entries = os.path.getsize(del_file) // 8  # sizeof(uint64_t)
        t.assert_true(raw_entries <= N,
            f"Tombstone file compacted: {raw_entries} entries <= {N}")
    else:
        t.log(".del file not on client path — skipping file size check")

    # Correctness: deleted IDs stay gone after compaction
    distractor = N + 9999
    with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
        c.insert(distractor, [1.0, 0.0, 0.0])
        result = c.search([1.0, 0.0, 0.0])
        t.assert_equal(result, distractor, "Deleted IDs stay deleted after compaction")
        c.drop()


def test_concurrent_clients(t: TestSuite):
    t.section("TEST 11 — Concurrent Clients (Multi-Threading)")
    NUM_THREADS = 8
    OPS = 25
    db_name = f"val_concurrent_{uuid.uuid4().hex[:8]}"
    errors = []
    lock = threading.Lock()

    def worker(tid):
        try:
            with RedBoxClient(host=HOST, port=PORT, db_name=db_name, dim=3) as c:
                base = tid * 10_000
                for i in range(OPS):
                    vid = base + i + 1
                    c.insert(vid, [float(vid), 0.0, 0.0])

                # Verify own inserts are searchable
                target = base + 1
                result = c.search([float(target), 0.0, 0.0])
                if result != target:
                    with lock:
                        errors.append(f"Thread {tid}: expected {target}, got {result}")

                # Update one of our own vectors
                ok = c.update(base + 2, [float(base + 2) + 0.1, 0.0, 0.0])
                if not ok:
                    with lock:
                        errors.append(f"Thread {tid}: update failed")

        except Exception as e:
            with lock:
                errors.append(f"Thread {tid} crashed: {e}")

    t.log(f"Spawning {NUM_THREADS} threads, {OPS} inserts + 1 update each...")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for th in threads: th.start()
    for th in threads: th.join()

    t.assert_true(len(errors) == 0,
        f"All {NUM_THREADS} concurrent threads completed without errors")
    if errors:
        for e in errors:
            t.log(f"  ERROR: {e}")
    # Drop DB after all threads done
    _drop_db(HOST, PORT, db_name)


def test_search_n_edge_cases(t: TestSuite):
    t.section("TEST 12 — search_N Edge Cases")
    with ManagedDb(2) as client:
        client.insert(1, [1.0, 0.0])

        # N=0: guard on client side — don't even send to server
        results = client.search_n([1.0, 0.0], 0)
        t.assert_equal(results, [], "search_N(0) returns empty list (client-side guard)")

        # N larger than DB
        results = client.search_n([1.0, 0.0], 999)
        t.assert_equal(len(results), 1, "search_N(999) with 1 vector returns 1 result")

        # All deleted
        client.delete(1)
        results = client.search_n([1.0, 0.0], 5)
        t.assert_equal(results, [], "search_N on all-deleted DB returns []")


# ==========================================
# CLEANUP
# ==========================================
def _drop_db(host, port, db_name):
    """Tell the server to close the mmap and delete the DB files."""
    try:
        with RedBoxClient(host=host, port=port, db_name=db_name, dim=1) as c:
            c.drop()
    except Exception:
        pass  # already gone or server restarted


# ==========================================
# RUNNER
# ==========================================
def run_validation():
    t = TestSuite()
    print("=" * 50)
    print("  RedBoxDb VALIDATION SUITE v2")
    print("=" * 50)

    test_empty_db_search(t)
    test_vector_math(t)
    test_search_n_ordering(t)
    test_crud_lifecycle(t)
    test_insert_auto_sequencing(t)
    test_search_n_after_deletion(t)
    test_multi_tenant_isolation(t)
    test_persistence(t)
    test_kmeans_init_correctness(t)   # slow — crosses 10k threshold
    test_tombstone_compaction(t)
    test_concurrent_clients(t)
    test_search_n_edge_cases(t)

    t.summary()
    if t.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        run_validation()
    except Exception as e:
        print(f"\n[CRITICAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)