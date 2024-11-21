#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#include "utils.hpp"

using time_unit = std::micro;
const std::string time_unit_str = "μs";

void benchmarkRunner(const std::function<void()>& func, std::string name) {
    std::string title = "=== " + name + " Benchmark ===";
    std::cout << "\033[1;32m" << title << "\033[0m" << std::endl;

    constexpr int warmupRuns = 10;
    constexpr int benchmarkRuns = 100;

    std::cout << "Warming up ..." << std::endl;

    for (int i = 0; i < warmupRuns; i++) {
        func();
    }

    std::cout << "Running benchmark ..." << std::endl;

    std::vector<double> runTimes;
    runTimes.reserve(benchmarkRuns);
    for (int i = 0; i < benchmarkRuns; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, time_unit> duration = end - start;
        runTimes.push_back(duration.count());
    }

    double totalTime = std::accumulate(runTimes.begin(), runTimes.end(), 0, std::plus<double>());
    double meanTime = totalTime / benchmarkRuns;
    std::cout << "\e[1m" <<  "Avg. execution time: " << meanTime << time_unit_str << std::endl;

    double variance = 0.0;
    for (double time : runTimes) {
        variance += (time - meanTime) * (time - meanTime);
    }
    variance /= benchmarkRuns;
    double stdDevTime = std::sqrt(variance);
    std::cout << "\e[1m" <<  "std: " << stdDevTime << time_unit_str << std::endl;

    double minTime = *std::min_element(runTimes.begin(), runTimes.end());
    std::cout << "\e[1m" <<  "Fastest time: " << minTime << time_unit_str << std::endl;

    double maxTime = *std::max_element(runTimes.begin(), runTimes.end());
    std::cout << "\e[1m" <<  "Slowest time: " << maxTime << time_unit_str << std::endl;

    std::string tail(title.length(), '=');
    std::cout << "\033[1;32m" << tail << "\033[0m" << std::endl;
}

void matmul_2x2_2x2() {
    Vec2d<double> a {{1,2},{3,4}};
    Vec2d<double> b {{5,6},{7,8}};
    for (int i = 0; i < 100; i++) {
        a * b;
    }
}

int main() {
    benchmarkRunner(matmul_2x2_2x2, "matmul_2x2_2x2");

    return 0;
}