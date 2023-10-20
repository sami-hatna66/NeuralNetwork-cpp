#ifndef Loss_hpp
#define Loss_hpp

#include "Layer.hpp"
#include "utils.hpp"

#include <numeric>

namespace Loss {

template <typename T> class LossBase {
  protected:
    Vec2d<T> dInputs;
    T accumulatedLoss;
    int accumulatedCount;

  public:
    LossBase() {}
    T calculate(Vec2d<T> &output, Vec2d<T> &y);
    T calculateRegLoss(Layers::DenseLayer<T> &layer);
    T calculateAccumulatedLoss();
    void newPass();
    Vec2d<T> getDInputs();

    virtual std::vector<T> compute(Vec2d<T> &predictY, Vec2d<T> &actualY) = 0;
    virtual void backward(Vec2d<T> &dValues, Vec2d<T> &actualY) = 0;
};

template <typename T> class CategoricalCrossEntropy : public LossBase<T> {
  private:
    using LossBase<T>::dInputs;

  public:
    std::vector<T> compute(Vec2d<T> &predictY, Vec2d<T> &actualY) override;
    void backward(Vec2d<T> &dValues, Vec2d<T> &actualY) override;
};

// Used for Binary Logistic Regression networks (ie, classification into two)
template <typename T> class BinaryCrossEntropy : public LossBase<T> {
  private:
    using LossBase<T>::dInputs;

  public:
    std::vector<T> compute(Vec2d<T> &predictY, Vec2d<T> &actualY) override;
    void backward(Vec2d<T> &dValues, Vec2d<T> &actualY) override;
};

template <typename T> class MeanSquaredError : public LossBase<T> {
  private:
    using LossBase<T>::dInputs;

  public:
    std::vector<T> compute(Vec2d<T> &predictY, Vec2d<T> &actualY) override;
    void backward(Vec2d<T> &dValues, Vec2d<T> &actualY) override;
};

template <typename T> class MeanAbsoluteError : public LossBase<T> {
  private:
    using LossBase<T>::dInputs;

  public:
    std::vector<T> compute(Vec2d<T> &predictY, Vec2d<T> &actualY) override;
    void backward(Vec2d<T> &dValues, Vec2d<T> &actualY) override;
};

} // namespace Loss

#endif
