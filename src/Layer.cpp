#include "Layer.hpp"

namespace Layers {

// FOR TESTING
// template <typename T>
// DenseLayer<T>::DenseLayer(int numInputs, int numNeurons, int num, T
// pWeightRegularizerL1,
//                 T pWeightRegularizerL2, T pBiasRegularizerL1,
//                 T pBiasRegularizerL2)
//     : weightRegularizerL1{pWeightRegularizerL1},
//       weightRegularizerL2{pWeightRegularizerL2},
//       biasRegularizerL1{pBiasRegularizerL1}, biasRegularizerL2{
//                                                  pBiasRegularizerL2} {
//     for (int i = 0; i < numInputs; i++) {
//         weightMomentums.push_back(std::vector<T>(numNeurons, 0.0));
//         weightCache.push_back(weightMomentums[i]);
//     }

//     if (num == 0) {
//         weights = {{{-1.30652683e-02, 1.65813062e-02,  -1.18164043e-03,
//                      -6.80178218e-03, 6.66383095e-03,  -4.60719783e-03,
//                      -1.33425845e-02, -1.34671740e-02, 6.93773152e-03,
//                      -1.59573427e-03, -1.33701565e-03, 1.07774371e-02,
//                      -1.12682581e-02, -7.30677694e-03, -3.84879787e-03,
//                      9.43515857e-04,  -4.21714503e-04, -2.86887190e-03,
//                      -6.16264006e-04, -1.07305276e-03, -7.19604362e-03,
//                      -8.12993012e-03, 2.74516339e-03,  -8.90915096e-03,
//                      -1.15735531e-02, -3.12292250e-03, -1.57667010e-03,
//                      2.25672331e-02,  -7.04700267e-03, 9.43260733e-03,
//                      7.47188320e-03,  -1.18894493e-02, 7.73252966e-03,
//                      -1.18388068e-02, -2.65917219e-02, 6.06319541e-03,
//                      -1.75589062e-02, 4.50934470e-03,  -6.84010889e-03,
//                      1.65955070e-02,  1.06850928e-02,  -4.53385804e-03,
//                      -6.87837601e-03, -1.21407732e-02, -4.40922612e-03,
//                      -2.80355476e-03, -3.64693534e-03, 1.56703859e-03,
//                      5.78521471e-03,  3.49654467e-03,  -7.64143933e-03,
//                      -1.43779144e-02, 1.36453183e-02,  -6.89449161e-03,
//                      -6.52293628e-03, -5.21189300e-03, -1.84306949e-02,
//                      -4.77973977e-03, -4.79655806e-03, 6.20358298e-03,
//                      6.98457099e-03,  3.77088909e-05,  9.31848306e-03,
//                      3.39964987e-03},
//                     {-1.56821116e-04, 1.60928175e-03,  -1.90653489e-03,
//                      -3.94849479e-03, -2.67733540e-03, -1.12801129e-02,
//                      2.80441693e-03,  -9.93123557e-03, 8.41631275e-03,
//                      -2.49458570e-03, 4.94949811e-04,  4.93836775e-03,
//                      6.43314468e-03,  -1.57062337e-02, -2.06903671e-03,
//                      8.80178902e-03,  -1.69810578e-02, 3.87280458e-03,
//                      -2.25556418e-02, -1.02250678e-02, 3.86305532e-04,
//                      -1.65671520e-02, -9.85510740e-03, -1.47183500e-02,
//                      1.64813492e-02,  1.64227746e-03,  5.67290280e-03,
//                      -2.22675106e-03, -3.53431748e-03, -1.61647405e-02,
//                      -2.91837356e-03, -7.61492178e-03, 8.57923925e-03,
//                      1.14110177e-02,  1.46657871e-02,  8.52551963e-03,
//                      -5.98653918e-03, -1.11589693e-02, 7.66663160e-03,
//                      3.56292794e-03,  -1.76853836e-02, 3.55481799e-03,
//                      8.14519823e-03,  5.89255884e-04,  -1.85053668e-03,
//                      -8.07648432e-03, -1.44653469e-02, 8.00297968e-03,
//                      -3.09114461e-03, -2.33466644e-03, 1.73272118e-02,
//                      6.84501091e-03,  3.70824989e-03,  1.42061792e-03,
//                      1.51999481e-02,  1.71958935e-02,  9.29505099e-03,
//                      5.82224596e-03,  -2.09460296e-02, 1.23721908e-03,
//                      -1.30106951e-03, 9.39532300e-04,  9.43046063e-03,
//                      -2.73967721e-02}}};
//     } else {
//         weights = {{{-5.69312042e-03, 2.69904337e-03, -4.66845511e-03},
//                     {-1.41690606e-02, 8.68963450e-03, 2.76871910e-03},
//                     {-9.71104577e-03, 3.14817182e-03, 8.21585674e-03},
//                     {5.29264616e-05, 8.00564792e-03, 7.82601768e-04},
//                     {-3.95228993e-03, -1.15942042e-02, -8.59307649e-04},
//                     {1.94292923e-03, 8.75832699e-03, -1.15107466e-03},
//                     {4.57415590e-03, -9.64611955e-03, -7.82629102e-03},
//                     {-1.10389292e-03, -1.05462847e-02, 8.20247829e-03},
//                     {4.63130325e-03, 2.79095769e-03, 3.38904094e-03},
//                     {2.02104356e-02, -4.68864199e-03, -2.20144130e-02},
//                     {1.99300190e-03, -5.06035401e-04, -5.17519051e-03},
//                     {-9.78829805e-03, -4.39189514e-03, 1.81338424e-03},
//                     {-5.02816681e-03, 2.41245367e-02, -9.60504357e-03},
//                     {-7.93117285e-03, -2.28861999e-02, 2.51484429e-03},
//                     {-2.01640651e-02, -5.39454632e-03, -2.75670527e-03},
//                     {-7.09727919e-03, 1.73887257e-02, 9.94394347e-03},
//                     {1.31913684e-02, -8.82418826e-03, 1.12859402e-02},
//                     {4.96000936e-03, 7.71405920e-03, 1.02943880e-02},
//                     {-9.08763241e-03, -4.24317596e-03, 8.62595998e-03},
//                     {-2.65561901e-02, 1.51332803e-02, 5.53132035e-03},
//                     {-4.57039627e-04, 2.20507639e-03, -1.02993520e-02},
//                     {-3.49943363e-03, 1.10028433e-02, 1.29802199e-02},
//                     {2.69622393e-02, -7.39246665e-04, -6.58552907e-03},
//                     {-5.14233951e-03, -1.01804184e-02, -7.78547488e-04},
//                     {3.82732414e-03, -3.42422776e-04, 1.09634679e-02},
//                     {-2.34215800e-03, -3.47450632e-03, -5.81268454e-03},
//                     {-1.63263455e-02, -1.56776775e-02, -1.17915794e-02},
//                     {1.30142802e-02, 8.95260274e-03, 1.37496404e-02},
//                     {-1.33221159e-02, -1.96862463e-02, -6.60056295e-03},
//                     {1.75818941e-03, 4.98690270e-03, 1.04797222e-02},
//                     {2.84279673e-03, 1.74266864e-02, -2.22605676e-03},
//                     {-9.13079176e-03, -1.68121830e-02, -8.88971332e-03},
//                     {2.42117955e-03, -8.88720248e-03, 9.36742499e-03},
//                     {1.41232759e-02, -2.36958694e-02, 8.64052307e-03},
//                     {-2.23960392e-02, 4.01499076e-03, 1.22487051e-02},
//                     {6.48561050e-04, -1.27968919e-02, -5.85431186e-03},
//                     {-2.61645438e-03, -1.82244775e-03, -2.02896819e-03},
//                     {-1.09882781e-03, 2.13480042e-03, -1.20857367e-02},
//                     {-2.42019817e-03, 1.51826115e-02, -3.84645420e-03},
//                     {-4.43836069e-03, 1.07819736e-02, -2.55918447e-02},
//                     {1.18137859e-02, -6.31903764e-03, 1.63928559e-03},
//                     {9.63213562e-04, 9.42468084e-03, -2.67594750e-03},
//                     {-6.78025745e-03, 1.29784578e-02, -2.36417390e-02},
//                     {2.03341813e-04, -1.34792542e-02, -7.61573343e-03},
//                     {2.01125666e-02, -4.45954269e-04, 1.95069693e-03},
//                     {-1.78156272e-02, -7.29044667e-03, 1.96557399e-03},
//                     {3.54757695e-03, 6.16886560e-03, 8.62789893e-05},
//                     {5.27004153e-03, 4.53781895e-03, -1.82974041e-02},
//                     {3.70057212e-04, 7.67902425e-03, 5.89879788e-03},
//                     {-3.63858812e-03, -8.05626530e-03, -1.11831184e-02},
//                     {-1.31054013e-03, 1.13307983e-02, -1.95180420e-02},
//                     {-6.59891730e-03, -1.13980239e-02, 7.84957502e-03},
//                     {-5.54309599e-03, -4.70637623e-03, -2.16949568e-03},
//                     {4.45393240e-03, -3.92388972e-03, -3.04614305e-02},
//                     {5.43311890e-03, 4.39042924e-03, -2.19541020e-03},
//                     {-1.08403657e-02, 3.51780117e-03, 3.79235530e-03},
//                     {-4.70032869e-03, -2.16731476e-03, -9.30156466e-03},
//                     {-1.78589090e-03, -1.55042931e-02, 4.17318800e-03},
//                     {-9.44368448e-03, 2.38103140e-03, -1.40596293e-02},
//                     {-5.90057671e-03, -1.10489398e-03, -1.66069977e-02},
//                     {1.15147873e-03, -3.79147544e-03, -1.74235608e-02},
//                     {-1.30324280e-02, 6.05120044e-03, 8.95555969e-03},
//                     {-1.31908641e-03, 4.04761825e-03, 2.23843544e-03},
//                     {3.29622976e-03, 1.28598399e-02, -1.50699839e-02}}};
//     }

//     biases.push_back(std::vector<T>(numNeurons, 0.0));
//     biasMomentums.push_back(biases[0]);
//     biasCache.push_back(biases[0]);
//     dBiases.push_back({});
// }

template <typename T>
DenseLayer<T>::DenseLayer(int numInputs, int numNeurons, T pWeightRegularizerL1,
                          T pWeightRegularizerL2, T pBiasRegularizerL1,
                          T pBiasRegularizerL2)
    : weightRegularizerL1{pWeightRegularizerL1},
      weightRegularizerL2{pWeightRegularizerL2},
      biasRegularizerL1{pBiasRegularizerL1}, biasRegularizerL2{
                                                 pBiasRegularizerL2} {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<T> distr(0, 1);
    weights.resize(numInputs, std::vector<T>(numNeurons));
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numNeurons; j++) {
            weights[i][j] = 0.01 * distr(gen);
        }
        weightMomentums.push_back(std::vector<T>(numNeurons, 0.0));
        weightCache.push_back(weightMomentums[i]);
    }

    biases.push_back(std::vector<T>(numNeurons, 0.0));
    biasMomentums.push_back(biases[0]);
    biasCache.push_back(biases[0]);
    dBiases.push_back({});
}

template <typename T> void DenseLayer<T>::compute(const Vec2d<T> &pInputs) {
    inputs = pInputs;
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

    dInputs = dValues * transpose(weights);
}

template <typename T> Vec2d<T> &DenseLayer<T>::getOutput() { return output; }

template <typename T> Vec2d<T> &DenseLayer<T>::getWeights() { return weights; }

template <typename T> Vec2d<T> &DenseLayer<T>::getBiases() { return biases; }

template <typename T> Vec2d<T> &DenseLayer<T>::getDWeights() {
    return dWeights;
}

template <typename T> Vec2d<T> &DenseLayer<T>::getDBiases() { return dBiases; }

template <typename T> Vec2d<T> &DenseLayer<T>::getDInputs() { return dInputs; }

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

template <typename T> DropoutLayer<T>::DropoutLayer(T pRate) {
    rate = 1.0 - pRate;
}

template <typename T> void DropoutLayer<T>::compute(const Vec2d<T> &pInputs) {
    inputs = pInputs;

    mask = Vec2d<T>(inputs.size(), std::vector<T>(inputs[0].size()));
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::binomial_distribution<int> distr(1, rate);
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask[i].size(); j++) {
            mask[i][j] = distr(gen);
        }
    }
    mask = mask / rate;

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

template <typename T> Vec2d<T> DropoutLayer<T>::getOutput() { return output; }

template <typename T> Vec2d<T> DropoutLayer<T>::getDInputs() { return dInputs; }

// Explicit instantiations
template class DenseLayer<double>;
template class DenseLayer<float>;

template class DropoutLayer<double>;
template class DropoutLayer<float>;

} // namespace Layers
