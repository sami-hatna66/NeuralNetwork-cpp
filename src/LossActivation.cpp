#include "LossActivation.hpp"

namespace LossActivation {

template <typename T> SoftmaxCCE<T>::SoftmaxCCE() : loss{}, activation{} {}

template <typename T>
T SoftmaxCCE<T>::compute(Vec2d<T> inputs, Vec2d<T> actualY) {
    activation.compute(inputs);
    output = activation.getOutput();
    return loss.calculate(output, actualY);
}

template <typename T> T SoftmaxCCE<T>::calculateRegLoss(Layers::DenseLayer<T> &layer) {
    T regularizationLoss = 0.0;

    if (layer.getWeightRegularizerL1() > 0) {
        T weightSum = 0.0;
        for (int i = 0; i < layer.getWeights().size(); i++) {
            for (int j = 0; j < layer.getWeights()[i].size(); j++) {
                weightSum += std::abs(layer.getWeights()[i][j]);
            }
        }
        regularizationLoss += layer.getWeightRegularizerL1() * weightSum;
    }
    if (layer.getWeightRegularizerL2() > 0) {
        auto weightsSquared = power(layer.getWeights(), (T)2.0);
        T weightsSquaredSum = 0.0;
        for (int i = 0; i < weightsSquared.size(); i++) {
            for (int j = 0; j < weightsSquared[i].size(); j++) {
                weightsSquaredSum += weightsSquared[i][j];
            }
        }
        regularizationLoss +=
            layer.getWeightRegularizerL2() * weightsSquaredSum;
    }
    if (layer.getBiasRegularizerL1() > 0) {
        T biasSum = 0.0;
        for (int i = 0; i < layer.getBiases().size(); i++) {
            for (int j = 0; j < layer.getBiases()[i].size(); j++) {
                biasSum += std::abs(layer.getBiases()[i][j]);
            }
        }
        regularizationLoss += layer.getBiasRegularizerL1() * biasSum;
    }
    if (layer.getBiasRegularizerL2() > 0) {
        auto biasesSquared = power(layer.getBiases(), (T)2.0);
        T biasesSquaredSum = 0.0;
        for (int i = 0; i < biasesSquared.size(); i++) {
            for (int j = 0; j < biasesSquared[i].size(); j++) {
                biasesSquaredSum += biasesSquared[i][j];
            }
        }
        regularizationLoss += layer.getBiasRegularizerL2() * biasesSquaredSum;
    }

    return regularizationLoss;
}

template <typename T>
void SoftmaxCCE<T>::backward(Vec2d<T> dValues, Vec2d<T> actualY) {
    int numSamples = dValues.size();

    Vec2d<T> maxIdxs;
    if (actualY.size() > 1) {
        maxIdxs.push_back({});
        for (const auto &row : actualY) {
            T maxVal = row[0];
            T maxIdx = 0;
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
        T y = maxIdxs[0][i];
        dInputs[i][y] -= 1;
    }

    dInputs = dInputs / (T)numSamples;
}

template <typename T> Vec2d<T> &SoftmaxCCE<T>::getOutput() { return output; }

template <typename T> Vec2d<T> &SoftmaxCCE<T>::getDInputs() { return dInputs; }

// Explicit instantiations
template class SoftmaxCCE<double>;
template class SoftmaxCCE<float>;

} // namespace LossActivation
