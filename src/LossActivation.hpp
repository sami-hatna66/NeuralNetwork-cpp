#ifndef LossActivation_hpp
#define LossActivation_hpp

#include "Loss.hpp"
#include "Activations.hpp"

namespace LossActivation {

class SoftmaxCCE {
private:
    Activations::Softmax activation;
    Loss::CategoricalCrossEntropy loss;
    DoubleVec2d output;
    DoubleVec2d dInputs;
public:
    SoftmaxCCE();
    double compute(DoubleVec2d inputs, DoubleVec2d actualY);
    void backward(DoubleVec2d dValues, DoubleVec2d actualY);
    DoubleVec2d& getOutput();
    DoubleVec2d& getDInputs();
};

}

#endif
