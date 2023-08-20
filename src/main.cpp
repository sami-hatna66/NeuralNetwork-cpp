#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "utils.hpp"
#include "Layer.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "LossActivation.hpp"
#include "Optimizers.hpp"

int main() {
    Layer<double> layer1(2, 64, 0);
    Activations::Relu<double> activation1;
    Layer<double> layer2(64, 3, 1);
    auto optimizer = Optimizers::Adam<double>(0.05, 0.0000005);
    LossActivation::SoftmaxCCE<double> lossActivation;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        layer1.compute(spiralDataX);
        activation1.compute(layer1.getOutput());
        layer2.compute(activation1.getOutput());
        
        auto loss = lossActivation.compute(layer2.getOutput(), spiralDataY);
        auto accuracy = calculateAccuracy<double>(lossActivation.getOutput(), spiralDataY);

        if (i % 100 == 0) {
            std::cout << "epoch: " << i
                    << std::fixed << std::setprecision(3)
                    << ", accuracy: " << accuracy
                    << std::fixed << std::setprecision(15)
                    << ", loss: " << loss
                    << std::fixed << std::setprecision(3)
                    << ", lr: " << optimizer.getCurrentLearningRate()
                    << ", time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() << " ms"
                    << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now(); 
        }

        lossActivation.backward(lossActivation.getOutput(), spiralDataY);
        layer2.backward(lossActivation.getDInputs());
        activation1.backward(layer2.getDInputs());
        layer1.backward(activation1.getDInputs());

        optimizer.setup();
        optimizer.updateParams(layer1);
        optimizer.updateParams(layer2);
        optimizer.finalize();
    }

    return 0;
}
