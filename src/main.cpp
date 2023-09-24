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
    Layers::DenseLayer<double> layer1{0, 1, 64};
    Activations::Relu<double> activation1;
    Layers::DenseLayer<double> layer2{1, 64, 64};
    Activations::Relu<double> activation2;
    Layers::DenseLayer<double> layer3{2, 64, 1};
    Activations::Linear<double> activation3;
    Loss::MeanSquaredError<double> loss;
    auto optimizer = Optimizers::Adam<double>{0.005, 0.001};

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        layer1.compute(sineDataX);
        activation1.compute(layer1.getOutput());
        layer2.compute(activation1.getOutput());
        activation2.compute(layer2.getOutput());
        layer3.compute(activation2.getOutput());
        activation3.compute(layer3.getOutput());

        auto dataLoss = loss.calculate(activation3.getOutput(), sineDataY);
        auto regLoss = loss.calculateRegLoss(layer1) +
                       loss.calculateRegLoss(layer2) +
                       loss.calculateRegLoss(layer3);
        auto lossVal = dataLoss + regLoss;
        auto accuracy =
            calculateAccuracy<double>(activation3.getOutput(), sineDataY);

        if (i % 100 == 0) {
            auto endTime = std::chrono::high_resolution_clock::now();
            std::cout << "epoch: " << i << std::fixed << std::setprecision(3)
                      << ", accuracy: " << accuracy << std::fixed << std::fixed
                      << std::setprecision(3) << ", loss: " << lossVal
                      << " (data loss: " << dataLoss
                      << ", reg loss: " << regLoss << ")" << std::fixed
                      << std::setprecision(3)
                      << ", lr: " << optimizer.getCurrentLearningRate()
                      << ", time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             endTime - startTime)
                             .count()
                      << " ms" << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
        }

        loss.backward(activation3.getOutput(), sineDataY);
        activation3.backward(loss.getDInputs());
        layer3.backward(activation3.getDInputs());
        activation2.backward(layer3.getDInputs());
        layer2.backward(activation2.getDInputs());
        activation1.backward(layer2.getDInputs());
        layer1.backward(activation1.getDInputs());

        optimizer.setup();
        optimizer.updateParams(layer1);
        optimizer.updateParams(layer2);
        optimizer.updateParams(layer3);
        optimizer.finalize();
    }

    return 0;
}
