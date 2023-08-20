#ifndef Optimizers_hpp
#define Optimizers_hpp

#include "utils.hpp"
#include "Layer.hpp"

namespace Optimizers {

class OptimizerBase {
protected:
    double learningRate;
    double currentLearningRate;
    double decay;
    double iterations;
public: 
    OptimizerBase(double pLearningRate, double PDecay);
    void setup();
    virtual void updateParams(Layer& layer) = 0;
    void finalize();

    double getCurrentLearningRate();
};

class StochasticGradientDescent : public OptimizerBase {
private:
    double momentum;
public:
    StochasticGradientDescent(double pLearningRate = 1.0, double pDecay = 0.0, double pMomentum = 0.0);
    void updateParams(Layer& layer) override;
};

class Adagrad : public OptimizerBase {
private:
    double epsilon;
public:
    Adagrad(double pLearningRate = 1.0, double pDecay = 0.0, double pEpsilon = 0.0000001);
    void updateParams(Layer& layer) override;
};

class RMSprop : public OptimizerBase {
private:
    double epsilon;
    double rho;
public:
    RMSprop(double pLearningRate = 0.001, double pDecay = 0.0, double pEpsilon = 0.0000001, double pRho = 0.9);
    void updateParams(Layer& layer) override;
};

class Adam : public OptimizerBase {
private:
    double epsilon;
    double beta1;
    double beta2;
public:
    Adam(double pLearningRate = 0.001, double pDecay = 0.0, double pEpsilon = 0.0000001, double pBeta1 = 0.9, double pBeta2 = 0.999);
    void updateParams(Layer& layer) override; 
};

}

#endif
