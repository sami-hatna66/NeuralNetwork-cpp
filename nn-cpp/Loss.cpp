#include "Loss.hpp"

namespace Loss {

template <typename T> T LossBase<T>::calculate(Vec2d<T> &output, Vec2d<T> &y) {
    auto sampleLosses = compute(output, y);

    T dataLoss = std::accumulate(sampleLosses.begin(), sampleLosses.end(), 0.0,
                                 std::plus<T>()) /
                 sampleLosses.size();

    accumulatedLoss += std::accumulate(sampleLosses.begin(), sampleLosses.end(),
                                       0.0, std::plus<T>());
    accumulatedCount += sampleLosses.size();

    return dataLoss;
}

template <typename T> T LossBase<T>::calculateAccumulatedLoss() {
    return accumulatedLoss / accumulatedCount;
}

template <typename T> void LossBase<T>::newPass() {
    accumulatedLoss = 0;
    accumulatedCount = 0;
}

template <typename T>
T LossBase<T>::calculateRegLoss(Layers::DenseLayer<T> &layer) {
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

template <typename T> Vec2d<T> LossBase<T>::getDInputs() { return dInputs; }

template <typename T>
std::vector<T> CategoricalCrossEntropy<T>::compute(Vec2d<T> &predictY,
                                                   Vec2d<T> &actualY) {
    int numSamples = predictY.size();

    // Trim data, prevents zero division
    T lowerLimit = 1.0E-7;
    T upperLimit = 1.0 - 1.0E-7;

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
        assert(pickedConfidences[i] != (T)0);
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

template <typename T>
std::vector<T> BinaryCrossEntropy<T>::compute(Vec2d<T> &predictY,
                                              Vec2d<T> &actualY) {
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

    Vec2d<T> sampleLosses =
        (T)0.0 - (eltwiseMult(actualY, log(predictY)) +
                  eltwiseMult(((T)1.0 - actualY), log((T)1.0 - predictY)));

    std::vector<T> result(sampleLosses.size());
    for (int i = 0; i < sampleLosses.size(); i++) {
        T sum = 0.0;
        for (int j = 0; j < sampleLosses[i].size(); j++) {
            sum += sampleLosses[i][j];
        }
        result[i] = sum / sampleLosses[i].size();
    }
    return result;
}

template <typename T>
void BinaryCrossEntropy<T>::backward(Vec2d<T> &dValues, Vec2d<T> &actualY) {
    T numSamples = dValues.size();
    T numLabels = dValues[0].size();

    // Trim data, prevents zero division
    T lowerLimit = 0.0000001;
    T upperLimit = 0.9999999;

    for (int i = 0; i < dValues.size(); i++) {
        for (int j = 0; j < dValues[i].size(); j++) {
            if (dValues[i][j] < lowerLimit)
                dValues[i][j] = lowerLimit;
            else if (dValues[i][j] > upperLimit)
                dValues[i][j] = upperLimit;
        }
    }

    dInputs =
        (T)0.0 - (actualY / dValues - ((T)1.0 - actualY) / ((T)1.0 - dValues)) /
                     numLabels;
    dInputs = dInputs / numSamples;
}

template <typename T>
std::vector<T> MeanSquaredError<T>::compute(Vec2d<T> &predictY,
                                            Vec2d<T> &actualY) {
    auto yAdjusted = power(actualY - predictY, (T)2.0);

    std::vector<T> result(yAdjusted.size());
    for (int i = 0; i < yAdjusted.size(); i++) {
        T sum = 0.0;
        for (int j = 0; j < yAdjusted[i].size(); j++) {
            sum += yAdjusted[i][j];
        }
        result[i] = sum / yAdjusted[i].size();
    }
    return result;
}

template <typename T>
void MeanSquaredError<T>::backward(Vec2d<T> &dValues, Vec2d<T> &actualY) {
    T numSamples = dValues.size();
    T numLabels = dValues[0].size();

    dInputs = (((actualY - dValues) * (T)-2.0) / numLabels) / numSamples;
}

template <typename T>
std::vector<T> MeanAbsoluteError<T>::compute(Vec2d<T> &predictY,
                                             Vec2d<T> &actualY) {
    auto yAdjusted = abs(actualY - predictY);

    std::vector<T> result(yAdjusted.size());
    for (int i = 0; i < yAdjusted.size(); i++) {
        T sum = 0.0;
        for (int j = 0; j < yAdjusted[i].size(); j++) {
            sum += yAdjusted[i][j];
        }
        result[i] = sum / yAdjusted[i].size();
    }
    return result;
}

template <typename T>
void MeanAbsoluteError<T>::backward(Vec2d<T> &dValues, Vec2d<T> &actualY) {
    T numSamples = dValues.size();
    T numLabels = dValues[0].size();

    Vec2d<T> signs(dValues.size(), std::vector<T>(dValues[0].size()));
    auto yAdjusted = actualY - dValues;
    for (int i = 0; i < signs.size(); i++) {
        for (int j = 0; j < signs[i].size(); j++) {
            signs[i][j] =
                yAdjusted[i][j] / std::sqrt(yAdjusted[i][j] * yAdjusted[i][j]);
        }
    }
    dInputs = (signs / numLabels) / numSamples;
}

// Explicit instantiations
template class LossBase<double>;
template class LossBase<float>;
template class CategoricalCrossEntropy<double>;
template class CategoricalCrossEntropy<float>;
template class BinaryCrossEntropy<double>;
template class BinaryCrossEntropy<float>;
template class MeanSquaredError<double>;
template class MeanSquaredError<float>;
template class MeanAbsoluteError<double>;
template class MeanAbsoluteError<float>;

} // namespace Loss
