#include "Optimizers.hpp"

namespace Optimizers {

template <typename T>
OptimizerBase<T>::OptimizerBase(T pLearningRate, T pDecay)
    : learningRate{pLearningRate}, currentLearningRate{pLearningRate},
      decay{pDecay}, iterations{0} {}

template <typename T> void OptimizerBase<T>::setup() {
    if (decay != 0) {
        // Exponential decaying
        currentLearningRate = learningRate * (1.0 / (1.0 + decay * iterations));
    }
}

template <typename T> void OptimizerBase<T>::finalize() { iterations += 1; }

template <typename T> T OptimizerBase<T>::getCurrentLearningRate() {
    return currentLearningRate;
}

template <typename T>
StochasticGradientDescent<T>::StochasticGradientDescent(T pLearningRate,
                                                        T pDecay, T pMomentum)
    : momentum{pMomentum}, OptimizerBase<T>{pLearningRate, pDecay} {}

template <typename T>
void StochasticGradientDescent<T>::updateParams(Layers::DenseLayer<T> &layer) {
    Vec2d<T> weightUpdates;
    Vec2d<T> biasUpdates;
    if (momentum != 0) {
        weightUpdates = (momentum * layer.getWeightMomentums()) -
                        (currentLearningRate * layer.getDWeights());
        layer.setWeightMomentums(weightUpdates);

        biasUpdates = (momentum * layer.getBiasMomentums()) -
                      (currentLearningRate * layer.getDBiases());
        layer.setBiasMomentums(biasUpdates);
    } else {
        weightUpdates = -currentLearningRate * layer.getDWeights();
        biasUpdates = -currentLearningRate * layer.getDBiases();
    }
    layer.setWeights(layer.getWeights() + weightUpdates);
    layer.setBiases(layer.getBiases() + biasUpdates);
}

template <typename T>
Adagrad<T>::Adagrad(T pLearningRate, T pDecay, T pEpsilon)
    : epsilon{pEpsilon}, OptimizerBase<T>{pLearningRate, pDecay} {}

// Adapts learning rate for each parameter based on historical gradients (stored in layer caches)
template <typename T>
void Adagrad<T>::updateParams(Layers::DenseLayer<T> &layer) {
    layer.setWeightCache(layer.getWeightCache() +
                         power<T>(layer.getDWeights(), 2.0));
    layer.setBiasCache(layer.getBiasCache() +
                       power<T>(layer.getDBiases(), 2.0));

    layer.setWeights(layer.getWeights() +
                     ((-currentLearningRate * layer.getDWeights()) /
                      (root(layer.getWeightCache()) + epsilon)));
    layer.setBiases(layer.getBiases() +
                    ((-currentLearningRate * layer.getDBiases()) /
                     (root(layer.getBiasCache()) + epsilon)));
}

template <typename T>
RMSprop<T>::RMSprop(T pLearningRate, T pDecay, T pEpsilon, T pRho)
    : epsilon{pEpsilon}, rho{pRho}, OptimizerBase<T>{pLearningRate, pDecay} {}

// RMSprop uses a more sophisticated method than Adagrad for adapting learning rates. Helps prevent aggressive lr decay
template <typename T>
void RMSprop<T>::updateParams(Layers::DenseLayer<T> &layer) {
    layer.setWeightCache((rho * layer.getWeightCache()) +
                         ((1 - rho) * power<T>(layer.getDWeights(), 2.0)));
    layer.setBiasCache((rho * layer.getBiasCache()) +
                       ((1 - rho) * power<T>(layer.getDBiases(), 2.0)));

    layer.setWeights(layer.getWeights() +
                     ((-currentLearningRate * layer.getDWeights()) /
                      (root(layer.getWeightCache()) + epsilon)));
    layer.setBiases(layer.getBiases() +
                    ((-currentLearningRate * layer.getDBiases()) /
                     (root(layer.getBiasCache()) + epsilon)));
}

template <typename T>
Adam<T>::Adam(T pLearningRate, T pDecay, T pEpsilon, T pBeta1, T pBeta2)
    : epsilon{pEpsilon}, beta1{pBeta1}, beta2{pBeta2},
      OptimizerBase<T>{pLearningRate, pDecay} {}

// RMSprop + momentums
template <typename T> void Adam<T>::updateParams(Layers::DenseLayer<T> &layer) {
    layer.setWeightMomentums(beta1 * layer.getWeightMomentums() +
                             (1 - beta1) * layer.getDWeights());
    layer.setBiasMomentums(beta1 * layer.getBiasMomentums() +
                           (1 - beta1) * layer.getDBiases());

    // Bias correction compensates for the fact that iterations starts as 0
    Vec2d<T> correctedWeightMomentums =
        layer.getWeightMomentums() / (1 - std::pow(beta1, iterations + 1));
    Vec2d<T> correctedBiasMomentums =
        layer.getBiasMomentums() / (1 - std::pow(beta1, iterations + 1));

    layer.setWeightCache(beta2 * layer.getWeightCache() +
                         (1 - beta2) * power<T>(layer.getDWeights(), 2.0));
    layer.setBiasCache(beta2 * layer.getBiasCache() +
                       (1 - beta2) * power<T>(layer.getDBiases(), 2.0));

    // Bias correction for cache
    Vec2d<T> correctedWeightCache =
        layer.getWeightCache() / (1 - std::pow(beta2, iterations + 1));
    Vec2d<T> correctedBiasCache =
        layer.getBiasCache() / (1 - std::pow(beta2, iterations + 1));

    layer.setWeights(layer.getWeights() +
                     (-currentLearningRate * correctedWeightMomentums /
                      (root(correctedWeightCache) + epsilon)));
    layer.setBiases(layer.getBiases() +
                    (-currentLearningRate * correctedBiasMomentums /
                     (root(correctedBiasCache) + epsilon)));
}

// Explicit instantiations
template class OptimizerBase<double>;
template class OptimizerBase<float>;
template class StochasticGradientDescent<double>;
template class StochasticGradientDescent<float>;
template class Adagrad<double>;
template class Adagrad<float>;
template class RMSprop<double>;
template class RMSprop<float>;
template class Adam<double>;
template class Adam<float>;

} // namespace Optimizers
