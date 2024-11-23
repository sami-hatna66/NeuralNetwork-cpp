#include <random>
#include <vector>
#include <algorithm>

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

class MatrixMulBenchFixture : public MatrixBinaryOpBenchFixture {
public:
    MatrixMulBenchFixture(int pR1, int pC1, int pR2, int pC2) : MatrixBinaryOpBenchFixture(pR1, pC1, pR2, pC2) {}

    void run() override {
        a * b;
    }
};

class MatrixDivBenchFixture : public MatrixBinaryOpBenchFixture {
public:
    MatrixDivBenchFixture(int pR1, int pC1, int pR2, int pC2) : MatrixBinaryOpBenchFixture(pR1, pC1, pR2, pC2) {}

    void run() override {
        a / b;
    }
};

class MatrixTransposeBenchFixture : public MatrixUnaryOpBenchFixture {
public:
    MatrixTransposeBenchFixture(int pR, int pC) : MatrixUnaryOpBenchFixture(pR, pC) {}

    void run() override {
        transpose(a);
    }
};

class MatrixMeanBenchFixture : public MatrixUnaryOpBenchFixture {
public:
    MatrixMeanBenchFixture(int pR, int pC) : MatrixUnaryOpBenchFixture(pR, pC) {}

    void run() override {
        mean(a);
    }
};

int main() {
    // Matrix multiplication benchmarks ----------------------------------------------------------------------------
    MatrixMulBenchFixture matrixMulBench_100x100_100x100 {100, 100, 100, 100};
    benchmarkRunner(&matrixMulBench_100x100_100x100, "matrix_mul_100x100_100x100");

    MatrixMulBenchFixture matrixMulBench_500x500_500x500 {500, 500, 500, 500};
    benchmarkRunner(&matrixMulBench_500x500_500x500, "matrix_mul_500x500_500x500");

    MatrixMulBenchFixture matrixMulBench_1000x1000_1000x1000 {1000, 1000, 1000, 1000};
    benchmarkRunner(&matrixMulBench_1000x1000_1000x1000, "matrix_mul_1000x1000_1000x1000");
    // -------------------------------------------------------------------------------------------------------------

    // Matrix division benchmarks ----------------------------------------------------------------------------------
    MatrixDivBenchFixture matrixDivBench_100x100_100x100 {100, 100, 100, 100};
    benchmarkRunner(&matrixDivBench_100x100_100x100, "matrix_div_100x100_100x100");

    MatrixDivBenchFixture matrixDivBench_500x500_500x500 {500, 500, 500, 500};
    benchmarkRunner(&matrixDivBench_500x500_500x500, "matrix_div_500x500_500x500");

    MatrixDivBenchFixture matrixDivBench_1000x1000_1000x1000 {1000, 1000, 1000, 1000};
    benchmarkRunner(&matrixDivBench_1000x1000_1000x1000, "matrix_div_1000x1000_1000x1000");
    // -------------------------------------------------------------------------------------------------------------

    // Transpose benchmarks ----------------------------------------------------------------------------------------
    MatrixTransposeBenchFixture matrixTransposeBench_100x100 {100, 100};
    benchmarkRunner(&matrixTransposeBench_100x100, "matrix_transpose_100x100");

    MatrixTransposeBenchFixture matrixTransposeBench_500x500 {500, 500};
    benchmarkRunner(&matrixTransposeBench_500x500, "matrix_transpose_500x500");

    MatrixTransposeBenchFixture matrixTransposeBench_1000x1000 {1000, 1000};
    benchmarkRunner(&matrixTransposeBench_1000x1000, "matrix_transpose_1000x1000");
    // -------------------------------------------------------------------------------------------------------------

    // Matrix mean benchmarks --------------------------------------------------------------------------------------
    MatrixMeanBenchFixture matrixMeanBench_100x100 {100, 100};
    benchmarkRunner(&matrixMeanBench_100x100, "matrix_mean_100x100");

    MatrixMeanBenchFixture matrixMeanBench_500x500 {500, 500};
    benchmarkRunner(&matrixMeanBench_500x500, "matrix_mean_500x500");

    MatrixMeanBenchFixture matrixMeanBench_1000x1000 {1000, 1000};
    benchmarkRunner(&matrixMeanBench_1000x1000, "matrix_mean_1000x1000");
    // -------------------------------------------------------------------------------------------------------------

    return 0;
}
