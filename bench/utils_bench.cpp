#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "bench_helpers.hpp"
#include "utils.hpp"

class MatrixOpBenchFixture : public BenchFixture {
protected:
    Vec2d<double> a;
    Vec2d<double> b;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int r1, c1, r2, c2;
    
public:
    MatrixOpBenchFixture(int pR1, int pC1, int pR2, int pC2) : BenchFixture(), gen(rd()), dis(0, 100) {
        r1 = pR1;
        c1 = pC1;
        r2 = pR2;
        c2 = pC2;
    }

    void setup() override {
        a.resize(r1, std::vector<double>(c1));
        for (auto &row : a) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        b.resize(r2, std::vector<double>(c2));
        for (auto &row : b) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }
    }
};

class MatMulBenchFixture : public MatrixOpBenchFixture {
public:
    MatMulBenchFixture(int pR1, int pC1, int pR2, int pC2) : MatrixOpBenchFixture(pR1, pC1, pR2, pC2) {}

    void run() override {
        a * b;
    }
};

int main() {
    MatMulBenchFixture matMulBench_50x50_50x50 {50, 50, 50, 50};
    benchmarkRunner(&matMulBench_50x50_50x50, "matmul_50x50_50x50");

    return 0;
}