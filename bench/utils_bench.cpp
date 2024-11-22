#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "bench_helpers.hpp"
#include "utils.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 100);

void setup_matmul(std::tuple<Vec2d<double>, Vec2d<double>> &params, int r1,
                  int c1, int r2, int c2) {
    auto &a = std::get<0>(params);
    a.resize(r1, std::vector<double>(c1));
    for (auto &row : a) {
        std::ranges::generate(row, [&]() { return dis(gen); });
    }

    auto &b = std::get<1>(params);
    b.resize(r2, std::vector<double>(c2));
    for (auto &row : b) {
        std::ranges::generate(row, [&]() { return dis(gen); });
    }
}

void bench_matmul(const Vec2d<double> &a, const Vec2d<double> &b) { a *b; }

int main() {
    benchmarkRunner<decltype(setup_matmul), decltype(bench_matmul),
                    Vec2d<double>, Vec2d<double>>(
        setup_matmul, bench_matmul, "matmul_50x50_50x50", 50, 50, 50, 50);

    return 0;
}