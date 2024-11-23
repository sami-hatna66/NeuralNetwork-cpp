#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "bench_helpers.hpp"
#include "utils.hpp"

class MatrixBinaryOpBenchFixture : public BenchFixture {
protected:
    Vec2d<double> a;
    Vec2d<double> b;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int r1, c1, r2, c2;
    
public:
    MatrixBinaryOpBenchFixture(int pR1, int pC1, int pR2, int pC2) : BenchFixture(), gen(rd()), dis(0, 100) {
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

class MatrixUnaryOpBenchFixture : public BenchFixture {
protected:
    Vec2d<double> a;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int r, c;

public:
    MatrixUnaryOpBenchFixture(int pR, int pC) : BenchFixture(), gen(rd()), dis(0, 100) {
        r = pR;
        c = pC;
    }

    void setup() override {
        a.resize(r, std::vector<double>(c));
        for (auto& row : a) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }
    }
};

class MatMulBenchFixture : public MatrixBinaryOpBenchFixture {
public:
    MatMulBenchFixture(int pR1, int pC1, int pR2, int pC2) : MatrixBinaryOpBenchFixture(pR1, pC1, pR2, pC2) {}

    void run() override {
        a * b;
    }
};

class MatDivBenchFixture : public MatrixBinaryOpBenchFixture {
public:
    MatDivBenchFixture(int pR1, int pC1, int pR2, int pC2) : MatrixBinaryOpBenchFixture(pR1, pC1, pR2, pC2) {}

    void run() override {
        a / b;
    }
};

class MatTransposeBenchFixture : public MatrixUnaryOpBenchFixture {
public:
    MatTransposeBenchFixture(int pR, int pC) : MatrixUnaryOpBenchFixture(pR, pC) {}

    void run() override {
        transpose(a);
    }
};

int main() {
    MatMulBenchFixture matMulBench_100x100_100x100 {100, 100, 100, 100};
    benchmarkRunner(&matMulBench_100x100_100x100, "mat_mul_100x100_100x100");

    MatMulBenchFixture matMulBench_500x500_500x500 {500, 500, 500, 500};
    benchmarkRunner(&matMulBench_500x500_500x500, "mat_mul_500x500_500x500");

    MatMulBenchFixture matMulBench_1000x1000_1000x1000 {1000, 1000, 1000, 1000};
    benchmarkRunner(&matMulBench_1000x1000_1000x1000, "mat_mul_1000x1000_1000x1000");

    MatDivBenchFixture matDivBench_100x100_100x100 {100, 100, 100, 100};
    benchmarkRunner(&matDivBench_100x100_100x100, "mat_div_100x100_100x100");

    MatDivBenchFixture matDivBench_500x500_500x500 {500, 500, 500, 500};
    benchmarkRunner(&matDivBench_500x500_500x500, "mat_div_500x500_500x500");

    MatDivBenchFixture matDivBench_1000x1000_1000x1000 {1000, 1000, 1000, 1000};
    benchmarkRunner(&matDivBench_1000x1000_1000x1000, "mat_div_1000x1000_1000x1000");

    MatTransposeBenchFixture matTransposeBench_100x100 {100, 100};
    benchmarkRunner(&matTransposeBench_100x100, "mat_transpose_100x100");

    MatTransposeBenchFixture matTransposeBench_500x500 {500, 500};
    benchmarkRunner(&matTransposeBench_500x500, "mat_transpose_500x500");

    MatTransposeBenchFixture matTransposeBench_1000x1000 {1000, 1000};
    benchmarkRunner(&matTransposeBench_1000x1000, "mat_transpose_1000x1000");

    return 0;
}
