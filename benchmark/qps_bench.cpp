#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include "redboxdb/engine.hpp"

const int         NUM_VECTORS   = 100'000;
const int         DIMENSIONS    = 128;
const int         TIMED_QUERIES = 5'000;
const std::string DB_FILE       = "qps_bench.db";

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

void cleanup() {
    if (std::filesystem::exists(DB_FILE))          std::filesystem::remove(DB_FILE);
    if (std::filesystem::exists(DB_FILE + ".del")) std::filesystem::remove(DB_FILE + ".del");
}

int main() {
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "Threads: " << num_threads << "\n";

    cleanup();
    auto* db = new CoreEngine::RedBoxVector(DB_FILE, DIMENSIONS, NUM_VECTORS);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < NUM_VECTORS; ++i) {
            std::vector<float> v(DIMENSIONS);
            for (auto& x : v) x = dis(rng);
            db->insert_auto(v);
        }
        std::cout << "Insert done.\n";
    }

    // Warmup: touch all float_block pages so they're resident before timing
    {
        std::cout << "Warming up pages...\n" << std::flush;
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        std::vector<float> warmup(DIMENSIONS);
        for (auto& x : warmup) x = dis(rng);
        for (int i = 0; i < 500; ++i)
            (void)db->search(warmup);
        std::cout << "Warmup done.\n" << std::flush;
    }

    std::cout << "Generating queries...\n" << std::flush;
    std::vector<std::vector<float>> queries(num_threads * TIMED_QUERIES);
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& q : queries) {
            q.resize(DIMENSIONS);
            for (auto& x : q) x = dis(rng);
        }
    }
    std::cout << "Queries ready. Launching " << num_threads << " threads...\n" << std::flush;

    std::vector<double> all_latencies(num_threads * TIMED_QUERIES);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto wall_start = Clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int base = t * TIMED_QUERIES;
            for (int i = 0; i < TIMED_QUERIES; ++i) {
                auto t0 = Clock::now();
                (void)db->search(queries[base + i]);
                auto t1 = Clock::now();
                all_latencies[base + i] = Ms(t1 - t0).count();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto wall_end = Clock::now();
    std::cout << "All done.\n" << std::flush;

    double wall_secs = std::chrono::duration<double>(wall_end - wall_start).count();
    long long total_q = (long long)num_threads * TIMED_QUERIES;
    double qps = total_q / wall_secs;

    std::sort(all_latencies.begin(), all_latencies.end());
    size_t n = all_latencies.size();
    double avg = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / n;

    std::cout << "\n--- RESULTS ---\n";
    std::cout << "Total queries : " << total_q << "\n";
    std::cout << "Wall time     : " << std::fixed << std::setprecision(3) << wall_secs << " s\n";
    std::cout << "QPS           : " << std::setprecision(1) << qps << "\n";
    std::cout << "Avg  : " << std::setprecision(3) << avg                  << " ms\n";
    std::cout << "P50  : " << all_latencies[n*50/100]                      << " ms\n";
    std::cout << "P95  : " << all_latencies[n*95/100]                      << " ms\n";
    std::cout << "P99  : " << all_latencies[n*99/100]                      << " ms\n";
    std::cout << "Max  : " << all_latencies.back()                         << " ms\n";

    delete db;
    cleanup();
    return 0;
}