#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>
#include <functional>
#include <tuple>

#include "utils.hpp"

using time_unit = std::micro;
const std::string time_unit_str = "μs";

class BenchFixture {
public:
    BenchFixture() {}
    virtual void setup() = 0;
    virtual void run() = 0;
};

void benchmarkRunner(BenchFixture* benchFixture, std::string name) {
    std::string title = "=== " + name + " Benchmark ===";
    std::cout << "\033[1;32m" << title << "\033[0m" << std::endl;

    constexpr int warmupRuns = 10;
    constexpr int benchmarkRuns = 100;

    std::cout << "Warming up ..." << std::endl;

    for (int i = 0; i < warmupRuns; i++) {
        benchFixture->setup();
        benchFixture->run();
    }

    std::cout << "Running benchmark ..." << std::endl;

    std::vector<double> runTimes;
    runTimes.reserve(benchmarkRuns);
    for (int i = 0; i < benchmarkRuns; i++) {
        benchFixture->setup();

        auto start = std::chrono::high_resolution_clock::now();
        benchFixture->run();
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