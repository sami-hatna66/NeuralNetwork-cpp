#ifndef Accuracy_hpp
#define Accuracy_hpp

#include "utils.hpp"

namespace Accuracy {

template <typename T> class AccuracyBase {
  public:
    T calculate(Vec2d<T> &predictions, Vec2d<T> &actualY);
    virtual Vec2d<T> predict(Vec2d<T> &predictions, Vec2d<T> &actualY) = 0;
};

template <typename T> class CategoricalAccuracy : public AccuracyBase<T> {
  private:
    bool isBinary;

  public:
    CategoricalAccuracy() {}
    CategoricalAccuracy(bool pIsBinary);
    Vec2d<T> predict(Vec2d<T> &predictions, Vec2d<T> &actualY) override;
};

template <typename T> class RegressionAccuracy : public AccuracyBase<T> {
  private:
    T precision;

  public:
    RegressionAccuracy() {}
    RegressionAccuracy(Vec2d<T> &actualY);
    Vec2d<T> predict(Vec2d<T> &predictions, Vec2d<T> &actualY) override;
};

} // namespace Accuracy

#endif
