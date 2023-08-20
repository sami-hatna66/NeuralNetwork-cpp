#ifndef Activations_hpp
#define Activations_hpp

#include "utils.hpp"

#include <numeric>

namespace Activations {

class ActivationBase {
protected:
    DoubleVec2d inputs; 
    DoubleVec2d output;
    DoubleVec2d dInputs;
public:
    ActivationBase() {}
    virtual void compute(const DoubleVec2d& pInputs) = 0;
    virtual void backward(const DoubleVec2d& dValues) = 0;
    DoubleVec2d& getOutput();
    DoubleVec2d& getDInputs();
};

class Relu : public ActivationBase {
public: 
    Relu() {}
    void compute(const DoubleVec2d& pInputs) override;
    void backward(const DoubleVec2d& pValues) override;
};

class Softmax : public ActivationBase {
public: 
    Softmax() {}
    void compute(const DoubleVec2d& pInputs) override;
    void backward(const DoubleVec2d& pValues) override;
};

}

#endif
