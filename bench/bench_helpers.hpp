#ifndef bench_helpers_hpp
#define bench_helpers_hpp

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <functional>

#include "utils.hpp"

using time_unit = std::micro;

class BenchFixture {
public:
    BenchFixture() {}
    virtual void setup() = 0;
    virtual void run() = 0;
};

inline void benchmarkRunner(BenchFixture* benchFixture, std::string name) {
    std::string title = "=== " + name + " Benchmark ===";
    std::cout << "\033[1;32m" << title << "\033[0m" << std::endl;
    
    constexpr int warmupRuns = 10;
    constexpr int benchmarkRuns = 50;

    std::cout << "Warming up ..." << std::endl;

    std::cout.setstate(std::ios_base::failbit);
    for (int i = 0; i < warmupRuns; i++) {
        benchFixture->setup();
        benchFixture->run();
    }

    std::cout.clear();
    std::cout << "Running benchmark ..." << std::endl;

    std::vector<double> runTimes;
    runTimes.reserve(benchmarkRuns);
    std::cout.setstate(std::ios_base::failbit);
    for (int i = 0; i < benchmarkRuns; i++) {
        benchFixture->setup();

        auto start = std::chrono::high_resolution_clock::now();
        benchFixture->run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, time_unit> duration = end - start;
        runTimes.push_back(duration.count());
    }

    std::cout.clear();

    double totalTime = std::accumulate(runTimes.begin(), runTimes.end(), 0, std::plus<double>());
    double meanTime = totalTime / benchmarkRuns;
    std::cout << "\e[1m" <<  "Avg. execution time: " << (meanTime / 1000) << "ms" << std::endl;

    double variance = 0.0;
    for (double time : runTimes) {
        variance += (time - meanTime) * (time - meanTime);
    }
    variance /= benchmarkRuns;
    double stdDevTime = std::sqrt(variance);
    std::cout << "\e[1m" <<  "std: " << (stdDevTime / 1000) << "ms" << std::endl;

    double minTime = *std::min_element(runTimes.begin(), runTimes.end());
    std::cout << "\e[1m" <<  "Fastest time: "  << (minTime / 1000) << "ms" << std::endl;

    double maxTime = *std::max_element(runTimes.begin(), runTimes.end());
    std::cout << "\e[1m" <<  "Slowest time: "  << (maxTime / 1000) << "ms" << std::endl;

    std::string tail(title.length(), '=');
    std::cout << "\033[1;32m" << tail << "\033[0m" << std::endl;
}

template<typename T>
void populateVec2dWithOnes(Vec2d<T>& vec, int rows, int cols) {
    vec.resize(rows, std::vector<T>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vec[i][j] = 1;
        }
    }
}

#endif