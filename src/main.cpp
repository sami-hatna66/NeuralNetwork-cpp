#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Accuracy.hpp"
#include "Activations.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "LossActivation.hpp"
#include "Model.hpp"
#include "Optimizers.hpp"
#include "TestData.hpp"
#include "utils.hpp"

int main() {
    Model<double, Accuracy::CategoricalAccuracy, Optimizers::Adam,
          Loss::CategoricalCrossEntropy>
        m;

    Accuracy::CategoricalAccuracy<double> acc;
    m.setAccuracy(acc);

    Optimizers::Adam<double> opt(0.05, 0.00005);
    m.setOptimizer(opt);

    Loss::CategoricalCrossEntropy<double> loss;
    m.setLoss(loss);

    auto l1 = std::make_shared<Layers::DenseLayer<double>>(2, 512, 0, 0.0005, 0,
                                                           0.0005);
    m.addLayer(l1);

    auto a1 = std::make_shared<Activations::Relu<double>>();
    m.addLayer(a1);

    auto l2 = std::make_shared<Layers::DropoutLayer<double>>(0.1);
    m.addLayer(l2);

    auto l3 = std::make_shared<Layers::DenseLayer<double>>(512, 3);
    m.addLayer(l3);

    auto a2 = std::make_shared<Activations::Softmax<double>>();
    m.addLayer(a2);

    m.prepare();

    m.train(spiralDataX, spiralDataY, 100);

    return 0;
}
