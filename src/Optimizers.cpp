#include "Optimizers.hpp"

namespace Optimizers {

OptimizerBase::OptimizerBase(double pLearningRate, double pDecay) 
        : learningRate{pLearningRate}, currentLearningRate{pLearningRate}, decay{pDecay}, iterations{0} {}

void OptimizerBase::setup() {
    if (decay != 0) {
        currentLearningRate = learningRate * (1.0 / (1.0 + decay * iterations));
    }
}

void OptimizerBase::finalize() {
    iterations += 1;
}

double OptimizerBase::getCurrentLearningRate() {
    return currentLearningRate;
}

StochasticGradientDescent::StochasticGradientDescent(double pLearningRate, double pDecay, double pMomentum) 
        : momentum{pMomentum}, OptimizerBase{pLearningRate, pDecay} {}

void StochasticGradientDescent::updateParams(Layer& layer) {
    DoubleVec2d weightUpdates;
    DoubleVec2d biasUpdates;
    if (momentum != 0) {
        weightUpdates = (momentum * layer.getWeightMomentums()) - (currentLearningRate * layer.getDWeights());
        layer.setWeightMomentums(weightUpdates);

        biasUpdates = (momentum * layer.getBiasMomentums()) - (currentLearningRate * layer.getDBiases());
        layer.setBiasMomentums(biasUpdates);
    } else {
        weightUpdates = -currentLearningRate * layer.getDWeights();
        biasUpdates = -currentLearningRate * layer.getDBiases();
    }
    layer.setWeights(layer.getWeights() + weightUpdates);
    layer.setBiases(layer.getBiases() + biasUpdates);
}

Adagrad::Adagrad(double pLearningRate, double pDecay, double pEpsilon) 
        : epsilon{pEpsilon}, OptimizerBase{pLearningRate, pDecay} {}

void Adagrad::updateParams(Layer& layer) {
    layer.setWeightCache(layer.getWeightCache() + exp(layer.getDWeights(), 2.0));
    layer.setBiasCache(layer.getBiasCache() + exp(layer.getDBiases(), 2.0));

    layer.setWeights(layer.getWeights() + ((-currentLearningRate * layer.getDWeights()) / (root(layer.getWeightCache()) + epsilon)));
    layer.setBiases(layer.getBiases() + ((-currentLearningRate * layer.getDBiases()) / (root(layer.getBiasCache()) + epsilon)));
}

RMSprop::RMSprop(double pLearningRate, double pDecay, double pEpsilon, double pRho) 
        : epsilon{pEpsilon}, rho{pRho}, OptimizerBase{pLearningRate, pDecay} {}

void RMSprop::updateParams(Layer& layer) {
    layer.setWeightCache((rho * layer.getWeightCache()) + ((1 - rho) * exp(layer.getDWeights(), 2.0)));
    layer.setBiasCache((rho * layer.getBiasCache()) + ((1 - rho) * exp(layer.getDBiases(), 2.0)));

    layer.setWeights(layer.getWeights() + ((-currentLearningRate * layer.getDWeights()) / (root(layer.getWeightCache()) + epsilon)));
    layer.setBiases(layer.getBiases() + ((-currentLearningRate * layer.getDBiases()) / (root(layer.getBiasCache()) + epsilon)));
}

Adam::Adam(double pLearningRate, double pDecay, double pEpsilon, double pBeta1, double pBeta2)
        : epsilon{pEpsilon}, beta1{pBeta1}, beta2{pBeta2}, OptimizerBase{pLearningRate, pDecay} {}

void Adam::updateParams(Layer& layer) {
    layer.setWeightMomentums(beta1 * layer.getWeightMomentums() + (1 - beta1) * layer.getDWeights());
    layer.setBiasMomentums(beta1 * layer.getBiasMomentums() + (1 - beta1) * layer.getDBiases());

    DoubleVec2d correctedWeightMomentums = layer.getWeightMomentums() / (1 - std::pow(beta1, iterations + 1));
    DoubleVec2d correctedBiasMomentums = layer.getBiasMomentums() / (1 - std::pow(beta1, iterations + 1));

    layer.setWeightCache(beta2 * layer.getWeightCache() + (1 - beta2) * exp(layer.getDWeights(), 2.0));
    layer.setBiasCache(beta2 * layer.getBiasCache() + (1 - beta2) * exp(layer.getDBiases(), 2.0));

    DoubleVec2d correctedWeightCache = layer.getWeightCache() / (1 - std::pow(beta2, iterations + 1));
    DoubleVec2d correctedBiasCache = layer.getBiasCache() / (1 - std::pow(beta2, iterations + 1));

    layer.setWeights(layer.getWeights() + (-currentLearningRate * correctedWeightMomentums / (root(correctedWeightCache) + epsilon)));
    layer.setBiases(layer.getBiases() + (-currentLearningRate * correctedBiasMomentums / (root(correctedBiasCache) + epsilon)));
} 

}
