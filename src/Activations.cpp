#include "Activations.hpp"

namespace Activations {

DoubleVec2d& ActivationBase::getOutput() {
    return output;
}

DoubleVec2d& ActivationBase::getDInputs() {
    return dInputs;
}


void Relu::compute(const DoubleVec2d& pInputs) {
    inputs = pInputs;
    if (output.size() == 0) {
        output = DoubleVec2d(inputs.size(), std::vector<double>(inputs[0].size(), 0.0));
    }
    for (int i = 0; i < inputs.size(); i++) {
        for (int j = 0; j < inputs[i].size(); j++) {
            output[i][j] = std::max(0.0, inputs[i][j]);
        }
    }
}

void Relu::backward(const DoubleVec2d& pValues) {
    dInputs = pValues;
    for (int i = 0; i < dInputs.size(); i++) {
        for (int j = 0; j < dInputs[i].size(); j++) {
            dInputs[i][j] = inputs[i][j] <= 0 ? 0 : dInputs[i][j];
        }
    }
}

void Softmax::compute(const DoubleVec2d& pInputs) {
    inputs = pInputs;

    DoubleVec2d expValues = DoubleVec2d(inputs.size(), std::vector<double>(inputs[0].size(), 0.0));
    for (int i = 0; i < inputs.size(); i++) {
        double rowMax = *std::max_element(inputs[i].begin(), inputs[i].end());
        for (int j = 0; j < inputs[i].size(); j++) {
            expValues[i][j] = std::exp(inputs[i][j] - rowMax);
        }
    }

    if (output.size() == 0) {
        output = DoubleVec2d(expValues.size(), std::vector<double>(expValues[0].size(), 0.0));
    }
    for (int i = 0; i < expValues.size(); i++) {
        double rowSum = std::accumulate(expValues[i].begin(), expValues[i].end(), 0.0, std::plus<double>());
        for (int j = 0; j < expValues[i].size(); j++) {
            output[i][j] = expValues[i][j] / rowSum;
        }
    }
}

void Softmax::backward(const DoubleVec2d& pValues) {
    dInputs = DoubleVec2d(pValues.size(), std::vector<double>(pValues[0].size(), 0.0));

    DoubleVec2d diagFlat(output[0].size(), std::vector<double>(output[0].size(), 0.0));
    for (int i = 0; i < pValues.size(); i++) {
        auto singleOutput = output[i];
        // Candidate for optimisation
        DoubleVec2d reshapedSingleOutput;
        for (int j = 0; j < singleOutput.size(); j++) {
            reshapedSingleOutput.push_back({singleOutput[j]});
        }

        for (int j = 0; j < reshapedSingleOutput.size(); j++) {
            for (int k = 0; k < reshapedSingleOutput.size(); k++) {
                if (j == k) diagFlat[j][k] = reshapedSingleOutput[j][0];
                else diagFlat[j][k] = 0.0;
            }
        }

        DoubleVec2d dotSingleOutput = reshapedSingleOutput * transpose(reshapedSingleOutput);

        DoubleVec2d jacobian = diagFlat - dotSingleOutput;

        std::vector<double> newRow(jacobian.size(), 0.0);
        for (int j = 0; j < jacobian.size(); j++) {
            for (int k = 0; k < jacobian[j].size(); k++) {
                newRow[j] += (pValues[i][k] * jacobian[j][k]);
            }
        }
        dInputs[i] = newRow;
    }
}

}
