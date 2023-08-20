#include "Loss.hpp"

namespace Loss {

template <typename T>
T CategoricalCrossEntropy<T>::calculate(Vec2d<T> &output, Vec2d<T> &y) {
    auto sampleLosses = compute(output, y);

    T dataLoss = std::accumulate(sampleLosses.begin(), sampleLosses.end(), 0.0,
                                 std::plus<T>()) /
                 sampleLosses.size();
    return dataLoss;
}

template <typename T>
std::vector<T> CategoricalCrossEntropy<T>::compute(Vec2d<T> &predictY,
                                                   Vec2d<T> &actualY) {
    int numSamples = predictY.size();

    // Trim data, prevents zero division
    T lowerLimit = 0.0000001;
    T upperLimit = 0.9999999;

    for (int i = 0; i < predictY.size(); i++) {
        for (int j = 0; j < predictY[i].size(); j++) {
            if (predictY[i][j] < lowerLimit)
                predictY[i][j] = lowerLimit;
            else if (predictY[i][j] > upperLimit)
                predictY[i][j] = upperLimit;
        }
    }

    std::vector<T> pickedConfidences;
    if (actualY.size() == 1) {
        for (int i = 0; i < numSamples; i++) {
            pickedConfidences.push_back(predictY[i][actualY[0][i]]);
        }
    } else {
        for (int i = 0; i < predictY.size(); i++) {
            for (int j = 0; j < predictY[i].size(); j++) {
                predictY[i][j] *= actualY[i][j];
            }
        }
        for (int i = 0; i < predictY.size(); i++) {
            pickedConfidences.push_back(std::accumulate(
                predictY[i].begin(), predictY[i].end(), 0.0, std::plus<T>()));
        }
    }

    for (int i = 0; i < pickedConfidences.size(); i++) {
        pickedConfidences[i] = -std::log(pickedConfidences[i]);
    }
    return pickedConfidences;
}

template <typename T>
void CategoricalCrossEntropy<T>::backward(Vec2d<T> &dValues,
                                          Vec2d<T> &actualY) {
    dInputs.clear();
    int numSamples = dValues.size();
    int numLabels = dValues[0].size();
    if (actualY.size() == 1) {
        Vec2d<T> oneHot;
        for (int i = 0; i < numSamples; i++) {
            std::vector<T> newRow(numLabels, 0.0);
            newRow[actualY[0][i]] = 1.0;
            oneHot.push_back(newRow);
        }
        actualY = oneHot;
    }
    for (int i = 0; i < dValues.size(); i++) {
        std::vector<T> newRow;
        for (int j = 0; j < dValues[i].size(); j++) {
            newRow.push_back((-actualY[i][j] / dValues[i][j]) / numSamples);
        }
        dInputs.push_back(newRow);
    }
}

template <typename T> Vec2d<T> CategoricalCrossEntropy<T>::getDInputs() {
    return dInputs;
}

// Explicit instantiations
template class CategoricalCrossEntropy<double>;
template class CategoricalCrossEntropy<float>;

} // namespace Loss
