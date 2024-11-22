#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>
#include <functional>
#include <tuple>

#include "utils.hpp"

using time_unit = std::micro;
const std::string time_unit_str = "μs";

template <typename SetupFunc, typename BenchFunc, typename... Params, typename... SetupArgs>
void benchmarkRunner(const SetupFunc& setupFunc, const BenchFunc& benchFunc, std::string name, SetupArgs&&... setupArgs) {
    std::string title = "=== " + name + " Benchmark ===";
    std::cout << "\033[1;32m" << title << "\033[0m" << std::endl;

    constexpr int warmupRuns = 10;
    constexpr int benchmarkRuns = 100;

    std::tuple<Params...> params;

    std::cout << "Warming up ..." << std::endl;

    for (int i = 0; i < warmupRuns; i++) {
        setupFunc(params, std::forward<SetupArgs>(setupArgs)...);
        std::apply(benchFunc, params);
    }

    std::cout << "Running benchmark ..." << std::endl;

    std::vector<double> runTimes;
    runTimes.reserve(benchmarkRuns);
    for (int i = 0; i < benchmarkRuns; i++) {
        setupFunc(params, std::forward<SetupArgs>(setupArgs)...);

        auto start = std::chrono::high_resolution_clock::now();
        std::apply(benchFunc, params);
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