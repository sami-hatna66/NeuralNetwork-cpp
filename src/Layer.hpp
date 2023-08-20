#ifndef Layer_hpp
#define Layer_hpp

#include "utils.hpp"

#include <random>

template <typename T>
class Layer {
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

public:
    Layer(int numInputs, int numNeurons, int num);
    Layer(int numInputs, int numNeurons);
    void compute(const Vec2d<T>& pInputs);
    void backward(const Vec2d<T>& dValues);

    Vec2d<T>& getOutput();
    Vec2d<T>& getWeights();
    Vec2d<T>& getBiases();
    Vec2d<T>& getDWeights();
    Vec2d<T>& getDBiases();
    Vec2d<T>& getDInputs();
    Vec2d<T>& getWeightMomentums();
    Vec2d<T>& getBiasMomentums();
    Vec2d<T>& getWeightCache();
    Vec2d<T>& getBiasCache();

    void setWeightMomentums(const Vec2d<T>& newWeightMomentums);
    void setBiasMomentums(const Vec2d<T>& newBiasMomentums);
    void setWeightCache(const Vec2d<T>& newWeightCache);
    void setBiasCache(const Vec2d<T>& newBiasCache);
    void setWeights(const Vec2d<T>& newWeights);
    void setBiases(const Vec2d<T>& newBiases);
};

#endif
 