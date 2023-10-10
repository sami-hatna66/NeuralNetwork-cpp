#ifndef Activations_hpp
#define Activations_hpp

#include "ModelLayer.hpp"
#include "utils.hpp"

#include <numeric>

namespace Activations {

template <typename T> class Relu : public ModelLayer<T> {
  private:
    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;

  public:
    Relu() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Softmax : public ModelLayer<T> {
  private:
    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;

  public:
    Softmax() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Sigmoid : public ModelLayer<T> {
  private:
    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;

  public:
    Sigmoid() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Linear : public ModelLayer<T> {
  private:
    using ModelLayer<T>::inputs;
    using ModelLayer<T>::output;
    using ModelLayer<T>::dInputs;

  public:
    Linear() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

} // namespace Activations

#endif
