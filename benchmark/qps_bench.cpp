#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include "redboxdb/engine.hpp"

const int         NUM_VECTORS   = 100'000;
const int         DIMENSIONS    = 128;
const int         WARMUP        = 200;
const int         TIMED_QUERIES = 5'000;   // per thread
const std::string DB_FILE       = "qps_bench.db";

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

void cleanup() {
    if (std::filesystem::exists(DB_FILE))          std::filesystem::remove(DB_FILE);
    if (std::filesystem::exists(DB_FILE + ".del")) std::filesystem::remove(DB_FILE + ".del");
}

std::vector<float> rand_vec(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dis(rng);
    return v;
}

int main() {
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());

    std::cout << "===============================================\n";
    std::cout << "   RedBoxDb PARALLEL QPS BENCHMARK\n";
    std::cout << "===============================================\n";
    std::cout << "Vectors    : " << NUM_VECTORS << "\n";
    std::cout << "Dimensions : " << DIMENSIONS  << "\n";
    std::cout << "Threads    : " << num_threads << "\n";
    std::cout << "===============================================\n\n";

    // --- PHASE 1: INSERT (single threaded) ---
    std::cout << "[1/2] Inserting " << NUM_VECTORS << " vectors...\n";
    cleanup();

    // Use a pointer so we can control lifetime
    auto* db = new CoreEngine::RedBoxVector(DB_FILE, DIMENSIONS, NUM_VECTORS);

    {
        std::mt19937 rng(42);
        auto t0 = Clock::now();
        for (int i = 0; i < NUM_VECTORS; ++i)
            db->insert_auto(rand_vec(DIMENSIONS, rng));
        auto t1 = Clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "   Done. " << (long long)(NUM_VECTORS / secs) << " vectors/sec\n\n";
    }

    // --- PHASE 2: PARALLEL SEARCH (shared instance, read-only) ---
    std::cout << "[2/2] PARALLEL SEARCH (" << num_threads << " threads x "
              << TIMED_QUERIES << " queries)\n";
    std::cout << "-----------------------------------------------\n";

    // Pre-generate all query vectors on main thread to avoid RNG contention
    std::mt19937 rng(99);
    std::vector<std::vector<float>> queries(num_threads * TIMED_QUERIES);
    for (auto& q : queries) q = rand_vec(DIMENSIONS, rng);

    // Warmup on main thread
    for (int i = 0; i < WARMUP; ++i)
        db->search(queries[i % queries.size()]);

    std::vector<std::vector<double>> per_thread_latencies(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::atomic<bool> go{false};
    std::atomic<int>  ready{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            auto& lat = per_thread_latencies[t];
            lat.reserve(TIMED_QUERIES);
            int base = t * TIMED_QUERIES;

            ready.fetch_add(1);
            while (!go.load()) { /* spin */ }

            for (int i = 0; i < TIMED_QUERIES; ++i) {
                auto t0 = Clock::now();
                (void)db->search(queries[base + i]);
                auto t1 = Clock::now();
                lat.push_back(Ms(t1 - t0).count());
            }
        });
    }

    // Wait for all threads ready, then fire
    while (ready.load() < num_threads) { /* spin */ }
    auto wall_start = Clock::now();
    go.store(true);

    for (auto& t : threads) t.join();
    auto wall_end = Clock::now();

    double wall_secs = std::chrono::duration<double>(wall_end - wall_start).count();
    long long total_q = (long long)num_threads * TIMED_QUERIES;
    double qps = total_q / wall_secs;

    // Aggregate latencies
    std::vector<double> all_lat;
    all_lat.reserve(total_q);
    for (auto& v : per_thread_latencies)
        all_lat.insert(all_lat.end(), v.begin(), v.end());
    std::sort(all_lat.begin(), all_lat.end());
    size_t n = all_lat.size();
    double avg = std::accumulate(all_lat.begin(), all_lat.end(), 0.0) / n;

    std::cout << "   Total queries : " << total_q << "\n";
    std::cout << "   Wall time     : " << std::fixed << std::setprecision(3) << wall_secs << " s\n";
    std::cout << "   QPS           : " << std::setprecision(1) << qps << " queries/sec\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "   Min  : " << all_lat.front()   << " ms\n";
    std::cout << "   Avg  : " << avg               << " ms\n";
    std::cout << "   P50  : " << all_lat[n*50/100] << " ms\n";
    std::cout << "   P95  : " << all_lat[n*95/100] << " ms\n";
    std::cout << "   P99  : " << all_lat[n*99/100] << " ms\n";
    std::cout << "   Max  : " << all_lat.back()    << " ms\n";

    delete db;
    cleanup();

    std::cout << "\n===============================================\n";
    std::cout << "   DONE\n";
    std::cout << "===============================================\n";
    return 0;
}