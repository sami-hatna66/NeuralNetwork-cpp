#include <random>
#include <vector>
#include <algorithm>

#include "bench_helpers.hpp"
#include "utils.hpp"
#include "Layer.hpp"

class DenseLayerBenchFixture : public BenchFixture {
protected:
    Layers::DenseLayer<double> layer;
    Vec2d<double> weights;
    Vec2d<double> biases;
    Vec2d<double> inputs;
    Vec2d<double> dValues;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int r, c;
    
public:
    DenseLayerBenchFixture(int pR, int pC) : BenchFixture(), gen(rd()), dis(0, 100), layer(pR, pC) {
        r = pR;
        c = pC;
    }

    void setup() override {
        weights.resize(r, std::vector<double>(c));
        for (auto &row : weights) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        biases.resize(1, std::vector<double>(c));
        for (auto &row : biases) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        inputs.resize(r, std::vector<double>(c));
        for (auto &row : inputs) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        dValues.resize(r, std::vector<double>(c));
        for (auto &row : dValues) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        layer.setWeights(weights);
        layer.setBiases(biases);
    
        layer.compute(inputs);
    }
};

class DenseLayerForwardBenchFixture : public DenseLayerBenchFixture {
public:
    DenseLayerForwardBenchFixture(int pR, int pC) : DenseLayerBenchFixture(pR, pC) {}

    void run() override {
        layer.compute(inputs);
    }
};

class DenseLayerBackpropBenchFixture : public DenseLayerBenchFixture {
public:
    DenseLayerBackpropBenchFixture(int pR, int pC) : DenseLayerBenchFixture(pR, pC) {}

    void run() override {
        layer.backward(dValues);
    }
};

class DropoutLayerBenchFixture : public BenchFixture {
protected:
    Layers::DropoutLayer<double> layer;
    Vec2d<double> inputs;
    Vec2d<double> dValues;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int r, c;
    
public:
    DropoutLayerBenchFixture(int pR, int pC, double pRate) : BenchFixture(), gen(rd()), dis(0, 100), layer(pRate) {
        r = pR;
        c = pC;
    }

    void setup() override {
        inputs.resize(r, std::vector<double>(c));
        for (auto &row : inputs) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }

        dValues.resize(r, std::vector<double>(c));
        for (auto &row : dValues) {
            std::ranges::generate(row, [&]() { return dis(gen); });
        }
    
        layer.compute(inputs);
    }
};

class DropoutLayerForwardBenchFixture : public DropoutLayerBenchFixture {
public:
    DropoutLayerForwardBenchFixture(int pR, int pC, double pRate) : DropoutLayerBenchFixture(pR, pC, pRate) {}

    void run() override {
        layer.compute(inputs);
    }
};

class DropoutLayerBackpropBenchFixture : public DropoutLayerBenchFixture {
public:
    DropoutLayerBackpropBenchFixture(int pR, int pC, double pRate) : DropoutLayerBenchFixture(pR, pC, pRate) {}

    void run() override {
        layer.backward(dValues);
    }
};

int main() {
    // dense forward propagation benchmarks -----------------------------------------------------------------------
    DenseLayerForwardBenchFixture denseFpropBench_100x100 {100, 100};
    benchmarkRunner(&denseFpropBench_100x100, "dense_fprop_100x100");

    DenseLayerForwardBenchFixture denseFpropBench_500x500 {500, 500};
    benchmarkRunner(&denseFpropBench_500x500, "dense_fprop_500x500");
    // ------------------------------------------------------------------------------------------------------------

    // dense backward propagation benchmarks ----------------------------------------------------------------------
    DenseLayerBackpropBenchFixture denseBackpropBench_100x100 {100, 100};
    benchmarkRunner(&denseBackpropBench_100x100, "dense_backprop_100x100");

    DenseLayerBackpropBenchFixture denseBackpropBench_500x500 {500, 500};
    benchmarkRunner(&denseBackpropBench_500x500, "dense_backprop_500x500");
    // ------------------------------------------------------------------------------------------------------------

    // dropout forward propagation benchmarks ---------------------------------------------------------------------
    DropoutLayerForwardBenchFixture dropoutFpropBench_100x100 {100, 100, 0.3};
    benchmarkRunner(&dropoutFpropBench_100x100, "dropout_fprop_100x100");

    DropoutLayerForwardBenchFixture dropoutFpropBench_500x500 {500, 500, 0.3};
    benchmarkRunner(&dropoutFpropBench_500x500, "dropout_fprop_500x500");
    // ------------------------------------------------------------------------------------------------------------

    // dropout backward propagation benchmarks --------------------------------------------------------------------
    DropoutLayerBackpropBenchFixture dropoutBackpropBench_100x100 {100, 100, 0.3};
    benchmarkRunner(&dropoutBackpropBench_100x100, "dropout_backprop_100x100");

    DropoutLayerBackpropBenchFixture dropoutBackpropBench_500x500 {500, 500, 0.3};
    benchmarkRunner(&dropoutBackpropBench_500x500, "dropout_backprop_500x500");
    // ------------------------------------------------------------------------------------------------------------

    return 0;
}