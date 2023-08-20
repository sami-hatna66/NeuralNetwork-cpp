#ifndef Layer_hpp
#define Layer_hpp

#include "utils.hpp"

#include <random>

class Layer {
private:
    DoubleVec2d inputs;
    DoubleVec2d weights;
    DoubleVec2d biases;
    DoubleVec2d output;

    DoubleVec2d dWeights;
    DoubleVec2d dBiases;
    DoubleVec2d dInputs;

    DoubleVec2d weightMomentums;
    DoubleVec2d biasMomentums;

    DoubleVec2d weightCache;
    DoubleVec2d biasCache;

public:
    Layer(int numInputs, int numNeurons, int num);
    Layer(int numInputs, int numNeurons);
    void compute(const DoubleVec2d& pInputs);
    void backward(const DoubleVec2d& dValues);

    DoubleVec2d& getOutput();
    DoubleVec2d& getWeights();
    DoubleVec2d& getBiases();
    DoubleVec2d& getDWeights();
    DoubleVec2d& getDBiases();
    DoubleVec2d& getDInputs();
    DoubleVec2d& getWeightMomentums();
    DoubleVec2d& getBiasMomentums();
    DoubleVec2d& getWeightCache();
    DoubleVec2d& getBiasCache();

    void setWeightMomentums(const DoubleVec2d& newWeightMomentums);
    void setBiasMomentums(const DoubleVec2d& newBiasMomentums);
    void setWeightCache(const DoubleVec2d& newWeightCache);
    void setBiasCache(const DoubleVec2d& newBiasCache);
    void setWeights(const DoubleVec2d& newWeights);
    void setBiases(const DoubleVec2d& newBiases);

};

#endif
 