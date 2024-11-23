#include "Layer.hpp"

namespace Layers {

template <typename T>
DenseLayer<T>::DenseLayer(int numInputs, int numNeurons, T pWeightRegularizerL1,
                          T pWeightRegularizerL2, T pBiasRegularizerL1,
                          T pBiasRegularizerL2)
    : weightRegularizerL1{pWeightRegularizerL1},
      weightRegularizerL2{pWeightRegularizerL2},
      biasRegularizerL1{pBiasRegularizerL1},
      biasRegularizerL2{pBiasRegularizerL2}, ModelLayer<T>{} {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<T> distr(0.0, 1.0);
    // Each neuron is connected to every neuron of the next layer
    // Every connection has an associated weight
    weights.resize(numInputs, std::vector<T>(numNeurons));
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNeurons; j++) {
            weights[i][j] = 0.01 * distr(gen);
        }
        weightMomentums.push_back(std::vector<T>(numNeurons, 0.0));
        weightCache.push_back(weightMomentums[i]);
    }

    // One bias per neuron, initialised as 0
    biases.push_back(std::vector<T>(numNeurons, 0.0));
    biasMomentums.push_back(biases[0]);
    biasCache.push_back(biases[0]);
    dBiases.push_back({});

    isTrainable = true;
}

template <typename T>
DenseLayer<T>::DenseLayer(Vec2d<T> pWeights, Vec2d<T> pBiases, T pWeightRegularizerL1,
                          T pWeightRegularizerL2, T pBiasRegularizerL1,
                          T pBiasRegularizerL2)
    : weightRegularizerL1{pWeightRegularizerL1},
      weightRegularizerL2{pWeightRegularizerL2},
      biasRegularizerL1{pBiasRegularizerL1},
      biasRegularizerL2{pBiasRegularizerL2}, ModelLayer<T>{} {
    weights = pWeights;
    biases = pBiases;
    biasMomentums.push_back(biases[0]);
    dBiases.push_back({});

    assert(weights[0].size() == biases[0].size());

    isTrainable = true;
}

template <typename T>
void DenseLayer<T>::compute(const Vec2d<T> &pInputs, LayerMode mode) {
    inputs = pInputs;
    // output = (input * weights) + biases
    output = inputs * weights;
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < biases[0].size(); j++) {
            output[i][j] += biases[0][j];
        }
    }
}

template <typename T> void DenseLayer<T>::backward(const Vec2d<T> &dValues) {
    dWeights = transpose(inputs) * dValues;
    dBiases.clear();
    std::vector<T> sums(dValues[0].size(), 0.0);
    for (int i = 0; i < dValues[0].size(); i++) {
        for (int j = 0; j < dValues.size(); j++) {
            sums[i] += dValues[j][i];
        }
    }
    dBiases.push_back(sums);

    if (weightRegularizerL1 > 0) {
        Vec2d<T> dL1(weights.size(), std::vector<T>(weights[0].size(), 1.0));
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                if (weights[i][j] < 0)
                    dL1[i][j] = -1;
            }
        }
        dWeights = dWeights + (weightRegularizerL1 * dL1);
    }
    if (weightRegularizerL2 > 0) {
        dWeights = dWeights + (2 * weightRegularizerL2 * weights);
    }
    if (biasRegularizerL1 > 0) {
        Vec2d<T> dL1(biases.size(), std::vector<T>(biases[0].size(), 1.0));
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases[i].size(); j++) {
                if (biases[i][j] < 0)
                    dL1[i][j] = -1;
            }
        }
        dBiases = dBiases + (biasRegularizerL1 * dL1);
    }
    if (biasRegularizerL2 > 0) {
        dBiases = dBiases + (2 * biasRegularizerL2 * biases);
    }

    // dInputs is the gradient - a vector of all the partial derivatives of the
    // function
    dInputs = dValues * transpose(weights);
}

template <typename T> Vec2d<T> &DenseLayer<T>::getWeights() { return weights; }

template <typename T> Vec2d<T> &DenseLayer<T>::getBiases() { return biases; }

template <typename T> Vec2d<T> &DenseLayer<T>::getDWeights() {
    return dWeights;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getDBiases() { return dBiases; }

template <typename T> T DenseLayer<T>::getWeightRegularizerL1() {
    return weightRegularizerL1;
}

template <typename T> T DenseLayer<T>::getWeightRegularizerL2() {
    return weightRegularizerL2;
}

template <typename T> T DenseLayer<T>::getBiasRegularizerL1() {
    return biasRegularizerL1;
}

template <typename T> T DenseLayer<T>::getBiasRegularizerL2() {
    return biasRegularizerL2;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getWeightMomentums() {
    return weightMomentums;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getBiasMomentums() {
    return biasMomentums;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getWeightCache() {
    return weightCache;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getBiasCache() {
    return biasCache;
}

template <typename T>
void DenseLayer<T>::setWeightMomentums(const Vec2d<T> &newWeightMomentums) {
    weightMomentums = newWeightMomentums;
}

template <typename T>
void DenseLayer<T>::setBiasMomentums(const Vec2d<T> &newBiasMomentums) {
    biasMomentums = newBiasMomentums;
}

template <typename T>
void DenseLayer<T>::setWeights(const Vec2d<T> &newWeights) {
    weights = newWeights;
}

template <typename T> void DenseLayer<T>::setBiases(const Vec2d<T> &newBiases) {
    biases = newBiases;
}

template <typename T>
void DenseLayer<T>::setWeightCache(const Vec2d<T> &newWeightCache) {
    weightCache = newWeightCache;
}

template <typename T>
void DenseLayer<T>::setBiasCache(const Vec2d<T> &newBiasCache) {
    biasCache = newBiasCache;
}

// Disables some neurons with p(1.0 - rate)
template <typename T>
DropoutLayer<T>::DropoutLayer(T pRate)
    : rate{(T)1.0 - pRate}, gen(rd()), distr(1, rate), ModelLayer<T>{} {}

template <typename T>
void DropoutLayer<T>::compute(const Vec2d<T> &pInputs, LayerMode mode) {
    inputs = pInputs;

    // Dropout only has an effect during training, not inference
    if (mode == LayerMode::Eval) {
        output = inputs;
    } else {
        if (mask.size() != inputs.size() ||
            mask[0].size() != inputs[0].size()) {
            mask = Vec2d<T>(inputs.size(), std::vector<T>(inputs[0].size()));
        }
        for (int i = 0; i < mask.size(); i++) {
            for (int j = 0; j < mask[i].size(); j++) {
                mask[i][j] = distr(gen) / rate;
            }
        }

        if (output.size() != inputs.size() ||
            output[0].size() != inputs[0].size()) {
            // avoids a copy in all but the first iteration
            output = Vec2d<T>(inputs.size(), std::vector<T>(inputs[0].size()));
        }
        for (int i = 0; i < output.size(); i++) {
            for (int j = 0; j < output[i].size(); j++) {
                output[i][j] = inputs[i][j] * mask[i][j];
            }
        }
    }
}

template <typename T> void DropoutLayer<T>::backward(const Vec2d<T> &dValues) {
    if (dInputs.size() != dValues.size() ||
        dInputs[0].size() != dValues[0].size()) {
        // avoids a copy in all but the first iteration
        dInputs = Vec2d<T>(dValues.size(), std::vector<T>(dValues[0].size()));
    }
    for (int i = 0; i < dInputs.size(); i++) {
        for (int j = 0; j < dInputs[i].size(); j++) {
            dInputs[i][j] = dValues[i][j] * mask[i][j];
        }
    }
}

template <typename T> InputLayer<T>::InputLayer() : ModelLayer<T>{} {}

template <typename T>
void InputLayer<T>::compute(const Vec2d<T> &pInputs, LayerMode mode) {
    output = pInputs;
}

template <typename T> void InputLayer<T>::backward(const Vec2d<T> &dValues) {}

// Explicit instantiations
template class DenseLayer<double>;
template class DenseLayer<float>;
template class DropoutLayer<double>;
template class DropoutLayer<float>;
template class InputLayer<double>;
template class InputLayer<float>;

} // namespace Layers
