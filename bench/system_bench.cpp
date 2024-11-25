#include <random>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>

#include "Model.hpp"
#include "bench_helpers.hpp"
#include "utils.hpp"

class ModelBenchFixture : public BenchFixture {
private:
    Vec2d<double> xTrain;
    Vec2d<double> yTrain;
    Vec2d<double> xTest;
    Vec2d<double> yTest;
    Model<double, Accuracy::CategoricalAccuracy, Optimizers::Adam, Loss::CategoricalCrossEntropy> m;

public:
    ModelBenchFixture() : BenchFixture() {
        Accuracy::CategoricalAccuracy<double> acc;
        m.setAccuracy(acc);

        Optimizers::Adam<double> opt(0.001, 0.001);
        m.setOptimizer(opt);

        Loss::CategoricalCrossEntropy<double> loss;
        m.setLoss(loss);

        auto l1 = std::make_shared<Layers::DenseLayer<double>>(784, 128);
        m.addLayer(l1);

        auto a1 = std::make_shared<Activations::Relu<double>>();
        m.addLayer(a1);

        auto l2 = std::make_shared<Layers::DenseLayer<double>>(128, 128);
        m.addLayer(l2);

        auto a2 = std::make_shared<Activations::Relu<double>>();
        m.addLayer(a2);

        auto l3 = std::make_shared<Layers::DenseLayer<double>>(128, 10);
        m.addLayer(l3);

        auto a3 = std::make_shared<Activations::Softmax<double>>();
        m.addLayer(a3);

        m.prepare();
    }

    void setup() override {
        populateVec2dWithOnes(xTrain, 60000, 784);
        populateVec2dWithOnes(yTrain, 1, 60000);
        populateVec2dWithOnes(xTest, 10000, 784);
        populateVec2dWithOnes(yTest, 1, 10000);
    }

    void run() override {
        m.train(std::make_unique<Vec2d<double>>(xTrain),
                std::make_unique<Vec2d<double>>(yTrain),
                std::make_unique<Vec2d<double>>(xTest),
                std::make_unique<Vec2d<double>>(yTest), 1, 128);
    }
};

int main() {
    // Benchmark one epoch of mnist model
    ModelBenchFixture mnistBench {};
    benchmarkRunner(&mnistBench, "mnist_model");

    return 0;
}