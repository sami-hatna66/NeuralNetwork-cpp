#include "Accuracy.hpp"

namespace Accuracy {

template <typename T>
T AccuracyBase<T>::calculate(Vec2d<T> &predictions, Vec2d<T> &actualY) {
    auto comparisons = predict(predictions, actualY);
    auto accuracy = mean(comparisons);

    accumulatedSum += std::accumulate(
        comparisons[0].begin(), comparisons[0].end(), 0.0, std::plus<T>());
    accumulatedCount += comparisons[0].size();

    return mean(predict(predictions, actualY));
}

template <typename T> T AccuracyBase<T>::calculateAccumulatedAcc() {
    return accumulatedSum / accumulatedCount;
}

template <typename T> void AccuracyBase<T>::newPass() {
    accumulatedSum = 0;
    accumulatedCount = 0;
}

template <typename T>
CategoricalAccuracy<T>::CategoricalAccuracy(bool pIsBinary)
    : isBinary{pIsBinary} {}

template <typename T>
Vec2d<T> CategoricalAccuracy<T>::predict(Vec2d<T> &predictions,
                                         Vec2d<T> &actualY) {
    Vec2d<T> adjY = {{}};
    if (!isBinary && actualY.size() > 1) {
        for (auto &row : actualY) {
            auto maxIter = std::max_element(row.begin(), row.end());
            adjY[0].push_back(std::distance(row.begin(), maxIter));
        }
    } else {
        adjY = actualY;
    }
    for (int i = 0; i < adjY[0].size(); i++) {
        adjY[0][i] = adjY[0][i] == predictions[0][i];
    }
    return adjY;
}

template <typename T>
RegressionAccuracy<T>::RegressionAccuracy(Vec2d<T> &actualY) {
    auto adjY = abs(actualY - mean(actualY));
    T stddevY = std::sqrt(mean(power(adjY, (T)2.0)));
    precision = stddevY / (T)250.0;
}

template <typename T>
Vec2d<T> RegressionAccuracy<T>::predict(Vec2d<T> &predictions,
                                        Vec2d<T> &actualY) {
    auto adjY = abs(predictions - actualY);
    for (int i = 0; i < adjY.size(); i++) {
        for (int j = 0; j < adjY[i].size(); j++) {
            adjY[i][j] = adjY[i][j] < precision;
        }
    }
    return adjY;
}

// Explicit instantiations
template class AccuracyBase<double>;
template class AccuracyBase<float>;
template class CategoricalAccuracy<double>;
template class CategoricalAccuracy<float>;
template class RegressionAccuracy<double>;
template class RegressionAccuracy<float>;

} // namespace Accuracy
