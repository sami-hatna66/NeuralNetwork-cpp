#ifndef Layer_hpp
#define Layer_hpp

#include "ModelLayer.hpp"
#include "utils.hpp"

#include <random>
#include <cassert>

namespace Layers {

template <typename T> class DenseLayer : public ModelLayer<T> {
  private:
    Vec2d<T> weights;
    Vec2d<T> biases;

    Vec2d<T> dWeights;
    Vec2d<T> dBiases;

    // Caches and Momentums used by optimizers
    Vec2d<T> weightMomentums;
    Vec2d<T> biasMomentums;

    Vec2d<T> weightCache;
    Vec2d<T> biasCache;

    // Regularization used by Loss to penalize models for having large weights
    // and biases (reduces generalization error)
    T weightRegularizerL1;
    T weightRegularizerL2;
    T biasRegularizerL1;
    T biasRegularizerL2;

    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;
    using ModelLayer<T>::isTrainable;

  public:
    DenseLayer(int numInputs, int numNeurons, T pWeightRegularizerL1 = 0,
               T pWeightRegularizerL2 = 0, T pBiasRegularizerL1 = 0,
               T pBiasRegularizerL2 = 0);
    DenseLayer(Vec2d<T> pWeights, Vec2d<T> pBiases, T pWeightRegularizerL1 = 0,
               T pWeightRegularizerL2 = 0, T pBiasRegularizerL1 = 0,
               T pBiasRegularizerL2 = 0);

    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &dValues) override;

    Vec2d<T> &getWeights();
    Vec2d<T> &getBiases();
    Vec2d<T> &getDWeights();
    Vec2d<T> &getDBiases();
    Vec2d<T> &getWeightMomentums();
    Vec2d<T> &getBiasMomentums();
    Vec2d<T> &getWeightCache();
    Vec2d<T> &getBiasCache();
    T getWeightRegularizerL1();
    T getWeightRegularizerL2();
    T getBiasRegularizerL1();
    T getBiasRegularizerL2();

    void setWeightMomentums(const Vec2d<T> &newWeightMomentums);
    void setBiasMomentums(const Vec2d<T> &newBiasMomentums);
    void setWeightCache(const Vec2d<T> &newWeightCache);
    void setBiasCache(const Vec2d<T> &newBiasCache);
    void setWeights(const Vec2d<T> &newWeights);
    void setBiases(const Vec2d<T> &newBiases);
};

template <typename T> class DropoutLayer : public ModelLayer<T> {
  private:
    Vec2d<T> mask;
    T rate;

    std::random_device rd;
    std::mt19937 gen;
    std::binomial_distribution<int> distr;

    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;

  public:
    DropoutLayer(T pRate);
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &dValues) override;
};

template <typename T> class InputLayer : public ModelLayer<T> {
  private:
    using ModelLayer<T>::output;

  public:
    InputLayer();
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &dValues) override;
};

} // namespace Layers

#endif
