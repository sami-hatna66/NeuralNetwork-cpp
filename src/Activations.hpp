#ifndef Activations_hpp
#define Activations_hpp

#include "utils.hpp"

#include <numeric>

namespace Activations {

template <typename T> class ActivationBase {
  protected:
    Vec2d<T> inputs;
    Vec2d<T> output;
    Vec2d<T> dInputs;

  public:
    ActivationBase() {}
    Vec2d<T> &getOutput();
    Vec2d<T> &getDInputs();

    virtual void compute(const Vec2d<T> &pInputs,
                         LayerMode mode = LayerMode::Training) = 0;
    virtual void backward(const Vec2d<T> &pValues) = 0;
    virtual Vec2d<T> predict(Vec2d<T> &outputs) = 0;
};

template <typename T> class Relu : public ActivationBase<T> {
  private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;

  public:
    Relu() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Softmax : public ActivationBase<T> {
  private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;

  public:
    Softmax() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Sigmoid : public ActivationBase<T> {
  private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;

  public:
    Sigmoid() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

template <typename T> class Linear : public ActivationBase<T> {
  private:
    using ActivationBase<T>::inputs;
    using ActivationBase<T>::output;
    using ActivationBase<T>::dInputs;

  public:
    Linear() {}
    void compute(const Vec2d<T> &pInputs,
                 LayerMode mode = LayerMode::Training) override;
    void backward(const Vec2d<T> &pValues) override;
    Vec2d<T> predict(Vec2d<T> &outputs) override;
};

} // namespace Activations

#endif
