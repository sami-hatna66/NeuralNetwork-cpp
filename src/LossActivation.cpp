#include "LossActivation.hpp"

namespace LossActivation {

SoftmaxCCE::SoftmaxCCE() : loss{}, activation{} {}

double SoftmaxCCE::compute(DoubleVec2d inputs, DoubleVec2d actualY) {
    activation.compute(inputs);
    output = activation.getOutput();
    return loss.calculate(output, actualY);
}

void SoftmaxCCE::backward(DoubleVec2d dValues, DoubleVec2d actualY) {
    int numSamples = dValues.size();

    DoubleVec2d maxIdxs;
    if (actualY.size() > 1) {
        maxIdxs.push_back({});
        for (const auto& row : actualY) {
            double maxVal = row[0];
            double maxIdx = 0;
            for (int i = 1; i < row.size(); i++) {
                if (row[i] > maxVal) {
                    maxVal = row[i];
                    maxIdx = i;
                }
            }
            maxIdxs[0].push_back(maxIdx);
        }
    } else {
        maxIdxs = actualY;
    }

    dInputs = dValues;

    for (int i = 0; i < numSamples; i++) {
        double y = maxIdxs[0][i];
        dInputs[i][y] -= 1;
    }

    dInputs = dInputs / (double) numSamples;
} 

DoubleVec2d& SoftmaxCCE::getOutput() {
    return output;
}

DoubleVec2d& SoftmaxCCE::getDInputs() {
    return dInputs;
}


}
