#include "Loss.hpp"

namespace Loss {

double CategoricalCrossEntropy::calculate(DoubleVec2d& output, DoubleVec2d& y) {
    auto sampleLosses = compute(output, y);

    double dataLoss =  std::accumulate(sampleLosses.begin(), sampleLosses.end(), 0.0, std::plus<double>()) / sampleLosses.size();
    return dataLoss;
}

std::vector<double> CategoricalCrossEntropy::compute(DoubleVec2d& predictY, DoubleVec2d& actualY) {
    int numSamples = predictY.size();

    // Trim data, prevents zero division
    double lowerLimit = 0.0000001;
    double upperLimit = 0.9999999;

    for (int i = 0; i < predictY.size(); i++) {
        for (int j = 0; j < predictY[i].size(); j++) {
            if (predictY[i][j] < lowerLimit) predictY[i][j] = lowerLimit;
            else if (predictY[i][j] > upperLimit) predictY[i][j] = upperLimit;
        }
    }

    std::vector<double> pickedConfidences;
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
            pickedConfidences.push_back(std::accumulate(predictY[i].begin(), predictY[i].end(), 0.0, std::plus<double>()));
        }
    }

    for (int i = 0; i < pickedConfidences.size(); i++) {
        pickedConfidences[i] = -std::log(pickedConfidences[i]);
    }
    return pickedConfidences;
}

void CategoricalCrossEntropy::backward(DoubleVec2d& dValues, DoubleVec2d& actualY) {
    dInputs.clear();
    int numSamples = dValues.size();
    int numLabels = dValues[0].size();
    if (actualY.size() == 1) {
        DoubleVec2d oneHot;
        for (int i = 0; i < numSamples; i++) {
            std::vector<double> newRow(numLabels, 0.0);
            newRow[actualY[0][i]] = 1.0;
            oneHot.push_back(newRow);
        }
        actualY = oneHot;
    }
    for (int i = 0; i < dValues.size(); i++) {
        std::vector<double> newRow;
        for (int j = 0; j < dValues[i].size(); j++) {
            newRow.push_back((-actualY[i][j] / dValues[i][j]) / numSamples);
        }
        dInputs.push_back(newRow);
    }
}

DoubleVec2d CategoricalCrossEntropy::getDInputs() {
    return dInputs;
}

}
