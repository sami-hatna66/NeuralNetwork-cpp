#include "Activations.hpp"

namespace Activations {

template <typename T> Vec2d<T> &ActivationBase<T>::getOutput() {
    return output;
}

template <typename T> Vec2d<T> &ActivationBase<T>::getDInputs() {
    return dInputs;
}

template <typename T> void Relu<T>::compute(const Vec2d<T> &pInputs) {
    inputs = pInputs;
    if (output.size() == 0) {
        output = Vec2d<T>(inputs.size(), std::vector<T>(inputs[0].size(), 0.0));
    }
    for (int i = 0; i < inputs.size(); i++) {
        for (int j = 0; j < inputs[i].size(); j++) {
            output[i][j] = std::max((T)0.0, inputs[i][j]);
        }
    }
}

template <typename T> void Relu<T>::backward(const Vec2d<T> &pValues) {
    dInputs = pValues;
    for (int i = 0; i < dInputs.size(); i++) {
        for (int j = 0; j < dInputs[i].size(); j++) {
            dInputs[i][j] = inputs[i][j] <= 0 ? 0 : dInputs[i][j];
        }
    }
}

template <typename T> void Softmax<T>::compute(const Vec2d<T> &pInputs) {
    inputs = pInputs;

    Vec2d<T> expValues =
        Vec2d<T>(inputs.size(), std::vector<T>(inputs[0].size(), 0.0));
    for (int i = 0; i < inputs.size(); i++) {
        T rowMax = *std::max_element(inputs[i].begin(), inputs[i].end());
        for (int j = 0; j < inputs[i].size(); j++) {
            expValues[i][j] = std::exp(inputs[i][j] - rowMax);
        }
    }

    if (output.size() == 0) {
        output = Vec2d<T>(expValues.size(),
                          std::vector<T>(expValues[0].size(), 0.0));
    }
    for (int i = 0; i < expValues.size(); i++) {
        T rowSum = std::accumulate(expValues[i].begin(), expValues[i].end(),
                                   0.0, std::plus<T>());
        for (int j = 0; j < expValues[i].size(); j++) {
            output[i][j] = expValues[i][j] / rowSum;
        }
    }
}

template <typename T> void Softmax<T>::backward(const Vec2d<T> &pValues) {
    dInputs = Vec2d<T>(pValues.size(), std::vector<T>(pValues[0].size(), 0.0));

    Vec2d<T> diagFlat(output[0].size(), std::vector<T>(output[0].size(), 0.0));
    for (int i = 0; i < pValues.size(); i++) {
        auto singleOutput = output[i];
        // Candidate for optimisation
        Vec2d<T> reshapedSingleOutput;
        for (int j = 0; j < singleOutput.size(); j++) {
            reshapedSingleOutput.push_back({singleOutput[j]});
        }

        for (int j = 0; j < reshapedSingleOutput.size(); j++) {
            for (int k = 0; k < reshapedSingleOutput.size(); k++) {
                if (j == k)
                    diagFlat[j][k] = reshapedSingleOutput[j][0];
                else
                    diagFlat[j][k] = 0.0;
            }
        }

        Vec2d<T> dotSingleOutput =
            reshapedSingleOutput * transpose(reshapedSingleOutput);

        Vec2d<T> jacobian = diagFlat - dotSingleOutput;

        std::vector<T> newRow(jacobian.size(), 0.0);
        for (int j = 0; j < jacobian.size(); j++) {
            for (int k = 0; k < jacobian[j].size(); k++) {
                newRow[j] += (pValues[i][k] * jacobian[j][k]);
            }
        }
        dInputs[i] = newRow;
    }
}

template <typename T> void Sigmoid<T>::compute(const Vec2d<T> &pInputs) {
    inputs = pInputs;
    output = (T)1.0 / (exp((T)0.0 - inputs) + (T)1.0);
}

template <typename T> void Sigmoid<T>::backward(const Vec2d<T> &pValues) {
    auto adjustedOutput = (T)1.0 - output;
    dInputs = eltwiseMult(pValues, eltwiseMult(adjustedOutput, output));
}

// Explicit instantiations
template class ActivationBase<double>;
template class ActivationBase<float>;
template class Relu<double>;
template class Relu<float>;
template class Softmax<double>;
template class Softmax<float>;
template class Sigmoid<double>;
template class Sigmoid<float>;

} // namespace Activations
