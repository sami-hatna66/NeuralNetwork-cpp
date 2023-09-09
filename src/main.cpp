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
    Layers::DenseLayer<double> layer1(2, 64, 0, 0.0005, 0, 0.0005);
    Activations::Relu<double> activation1;
    Layers::DenseLayer<double> layer2(64, 1);
    Activations::Sigmoid<double> activation2;
    Loss::BinaryCrossEntropy<double> loss;
    auto optimizer = Optimizers::Adam<double>(0.001, 0.0000005);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        layer1.compute(spiralDataX);
        activation1.compute(layer1.getOutput());
        layer2.compute(activation1.getOutput());
        activation2.compute(layer2.getOutput());

        auto dataLoss = loss.calculate(activation2.getOutput(), spiralDataY);
        auto regLoss =
            loss.calculateRegLoss(layer1) + loss.calculateRegLoss(layer2);
        auto lossVal = dataLoss + regLoss;
        auto accuracy =
            calculateAccuracy<double>(activation2.getOutput(), spiralDataY);

        if (i % 100 == 0) {
            std::cout << "epoch: " << i << std::fixed << std::setprecision(3)
                      << ", accuracy: " << accuracy << std::fixed << std::fixed
                      << std::setprecision(3) << ", loss: " << lossVal
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

        loss.backward(activation2.getOutput(), spiralDataY);
        activation2.backward(loss.getDInputs());
        layer2.backward(activation2.getDInputs());
        activation1.backward(layer2.getDInputs());
        layer1.backward(activation1.getDInputs());

        optimizer.setup();
        optimizer.updateParams(layer1);
        optimizer.updateParams(layer2);
        optimizer.finalize();
    }

    return 0;
}
