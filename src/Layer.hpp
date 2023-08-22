#ifndef Layer_hpp
#define Layer_hpp

#include "utils.hpp"

#include <random>

namespace Layers {

template <typename T> class DenseLayer {
  private:
    Vec2d<T> inputs;
    Vec2d<T> weights;
    Vec2d<T> biases;
    Vec2d<T> output;

    Vec2d<T> dWeights;
    Vec2d<T> dBiases;
    Vec2d<T> dInputs;

    Vec2d<T> weightMomentums;
    Vec2d<T> biasMomentums;

    Vec2d<T> weightCache;
    Vec2d<T> biasCache;

    T weightRegularizerL1;
    T weightRegularizerL2;
    T biasRegularizerL1;
    T biasRegularizerL2;

  public:
    DenseLayer(int numInputs, int numNeurons, int num, T pWeightRegularizerL1 = 0,
          T pWeightRegularizerL2 = 0, T pBiasRegularizerL1 = 0,
          T pBiasRegularizerL2 = 0);
    DenseLayer(int numInputs, int numNeurons, T pWeightRegularizerL1 = 0,
          T pWeightRegularizerL2 = 0, T pBiasRegularizerL1 = 0,
          T pBiasRegularizerL2 = 0);
    void compute(const Vec2d<T> &pInputs);
    void backward(const Vec2d<T> &dValues);

    Vec2d<T> &getOutput();
    Vec2d<T> &getWeights();
    Vec2d<T> &getBiases();
    Vec2d<T> &getDWeights();
    Vec2d<T> &getDBiases();
    Vec2d<T> &getDInputs();
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

template <typename T> 
class DropoutLayer {
  private:
    Vec2d<T> inputs;
    Vec2d<T> output;
    Vec2d<T> dInputs;
    Vec2d<T> mask;
    T rate;

  public:
    DropoutLayer(T pRate);
    void compute(const Vec2d<T> &pInputs);
    void backward(const Vec2d<T> &dValues);

    Vec2d<T> getOutput();
    Vec2d<T> getDInputs();
};

}

#endif