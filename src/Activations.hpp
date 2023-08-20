#ifndef Activations_hpp
#define Activations_hpp

#include "utils.hpp"

#include <numeric>

namespace Activations {

template <typename T>
class ActivationBase {
protected:
    Vec2d<T> inputs; 
    Vec2d<T> output;
    Vec2d<T> dInputs;
public:
    ActivationBase() {}
    Vec2d<T>& getOutput();
    Vec2d<T>& getDInputs();
};

template <typename T>
class Relu : public ActivationBase<T> {
private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;
public: 
    Relu() {}
    void compute(const Vec2d<T>& pInputs);
    void backward(const Vec2d<T>& pValues);
};

template <typename T>
class Softmax : public ActivationBase<T> {
private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;
public: 
    Softmax() {}
    void compute(const Vec2d<T>& pInputs);
    void backward(const Vec2d<T>& pValues);
};

}

#endif
