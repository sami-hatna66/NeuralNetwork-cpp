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

double calculateAccuracy(const DoubleVec2d& output, const DoubleVec2d& actualY) {
    auto argmax = [](const DoubleVec2d& inp) {
        std::vector<double> out(inp.size());
        for (int i = 0; i < inp.size(); i++) {
            auto maxIter = std::max_element(inp[i].begin(), inp[i].end());
            out[i] = std::distance(inp[i].begin(), maxIter);
        }
        return out;
    };

    std::vector<double> predictions = argmax(output);
    
    std::vector<double> y;
    if (actualY.size() != 1) {
        y = argmax(actualY);
    } else {
        for (int i = 0; i < actualY[0].size(); i++) {
            y.push_back(actualY[0][i]);
        }
    }

    std::vector<bool> isPredY(y.size());
    for (int i = 0; i < y.size(); i++) {
        isPredY[i] = y[i] == predictions[i];
    }
    return std::accumulate(isPredY.begin(), isPredY.end(), 0.0) / isPredY.size();
}

int main() {
    Layer layer1(2, 64, 0);
    Activations::Relu activation1;
    Layer layer2(64, 3, 1);
    auto optimizer = Optimizers::Adam(0.05, 0.0000005);
    LossActivation::SoftmaxCCE lossActivation;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        layer1.compute(spiralDataX);
        activation1.compute(layer1.getOutput());
        layer2.compute(activation1.getOutput());
        
        auto loss = lossActivation.compute(layer2.getOutput(), spiralDataY);
        auto accuracy = calculateAccuracy(lossActivation.getOutput(), spiralDataY);

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
