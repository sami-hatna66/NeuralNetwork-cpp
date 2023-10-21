#ifndef ModelLayer_hpp
#define ModelLayer_hpp

#include "utils.hpp"

// Base class shared by dense, input, dropout and activation layers
// Used in Model class as type for vector of pointers to layers
template <typename T> class ModelLayer {
  protected:
    Vec2d<T> inputs;
    Vec2d<T> output;
    Vec2d<T> dInputs;

    // Dense layers are trainable (they have weights), whereas activation, input
    // and dropout layers aren't
    bool isTrainable = false;

  public:
    ModelLayer() {}
    virtual void compute(const Vec2d<T> &pInputs,
                         LayerMode mode = LayerMode::Training) = 0;
    virtual void backward(const Vec2d<T> &dValues) = 0;
    Vec2d<T> &getOutput();
    Vec2d<T> &getDInputs();
    void setDInputs(Vec2d<T> &newDInputs);
    bool getIsTrainable();
    // Only implemented for activation layers
    virtual Vec2d<T> predict(Vec2d<T> &outputs);
};

#endif
