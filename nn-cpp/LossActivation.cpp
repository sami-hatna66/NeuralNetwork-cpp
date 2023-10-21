#include "LossActivation.hpp"

namespace LossActivation {

template <typename T> SoftmaxCCE<T>::SoftmaxCCE() : loss{}, activation{} {}

template <typename T>
T SoftmaxCCE<T>::compute(Vec2d<T> &inputs, Vec2d<T> &actualY) {
    activation.compute(inputs);
    output = activation.getOutput();
    return loss.calculate(output, actualY);
}

template <typename T>
void SoftmaxCCE<T>::backward(Vec2d<T> &dValues, Vec2d<T> &actualY) {
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
