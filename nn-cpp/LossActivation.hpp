#ifndef LossActivation_hpp
#define LossActivation_hpp

#include "Activations.hpp"
#include "Layer.cpp"
#include "Loss.hpp"

namespace LossActivation {

template <typename T> class SoftmaxCCE {
  private:
    Activations::Softmax<T> activation;
    Loss::CategoricalCrossEntropy<T> loss;
    Vec2d<T> output;
    Vec2d<T> dInputs;

  public:
    SoftmaxCCE();
    T compute(Vec2d<T> inputs, Vec2d<T> actualY);
    void backward(Vec2d<T> dValues, Vec2d<T> actualY);
    Vec2d<T> &getOutput();
    Vec2d<T> &getDInputs();
};

} // namespace LossActivation

#endif
