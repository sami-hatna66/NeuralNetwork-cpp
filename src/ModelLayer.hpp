#ifndef ModelLayer_hpp
#define ModelLayer_hpp

#include "utils.hpp"

template <typename T> class ModelLayer {
  protected:
    Vec2d<T> inputs;
    Vec2d<T> output;
    Vec2d<T> dInputs;

  public:
    ModelLayer() {}
    virtual void compute(const Vec2d<T> &pInputs,
                         LayerMode mode = LayerMode::Training) = 0;
    virtual void backward(const Vec2d<T> &dValues) = 0;
    Vec2d<T>& getOutput();
    Vec2d<T>& getDInputs();
};

#endif
