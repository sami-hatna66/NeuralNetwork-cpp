#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Activations.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "LossActivation.hpp"
#include "Optimizers.hpp"
#include "utils.hpp"

int main() {
    Layers::DenseLayer<double> layer1(2, 64, 0, 0, 0.0005, 0, 0.0005);
    Activations::Relu<double> activation1;
    Layers::DropoutLayer<double> dropout1(0.1);
    Layers::DenseLayer<double> layer2(64, 3, 1);
    LossActivation::SoftmaxCCE<double> lossActivation;
    auto optimizer = Optimizers::Adam<double>(0.05, 0.00005);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        layer1.compute(spiralDataX);
        activation1.compute(layer1.getOutput());
        dropout1.compute(activation1.getOutput());
        layer2.compute(dropout1.getOutput());

        auto dataLoss = lossActivation.compute(layer2.getOutput(), spiralDataY);
        auto regLoss = lossActivation.calculateRegLoss(layer1) +
                       lossActivation.calculateRegLoss(layer2);
        auto loss = dataLoss + regLoss;
        auto accuracy =
            calculateAccuracy<double>(lossActivation.getOutput(), spiralDataY);

        if (i % 100 == 0) {
            std::cout << "epoch: " << i << std::fixed << std::setprecision(3)
                      << ", accuracy: " << accuracy << std::fixed << std::fixed
                      << std::setprecision(3) << ", loss: " << loss
                      << " (data loss: " << dataLoss
                      << ", reg loss: " << regLoss << ")" << std::fixed
                      << std::setprecision(3)
                      << ", lr: " << optimizer.getCurrentLearningRate()
                      << ", time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::high_resolution_clock::now() -
                             startTime)
                             .count()
                      << " ms" << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
        }

        lossActivation.backward(lossActivation.getOutput(), spiralDataY);
        layer2.backward(lossActivation.getDInputs());
        dropout1.backward(layer2.getDInputs());
        activation1.backward(dropout1.getDInputs());
        layer1.backward(activation1.getDInputs());

        optimizer.setup();
        optimizer.updateParams(layer1);
        optimizer.updateParams(layer2);
        optimizer.finalize();
    }

    return 0;
}
