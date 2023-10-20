#ifndef Accuracy_hpp
#define Accuracy_hpp

#include "utils.hpp"

namespace Accuracy {

template <typename T> class AccuracyBase {
  private:
    T accumulatedSum;
    int accumulatedCount;

  public:
    T calculate(Vec2d<T> &predictions, Vec2d<T> &actualY);
    virtual Vec2d<T> predict(Vec2d<T> &predictions, Vec2d<T> &actualY) = 0;
    T calculateAccumulatedAcc();
    void newPass();
};

// For classification networks
template <typename T> class CategoricalAccuracy : public AccuracyBase<T> {
  private:
    bool isBinary;

  public:
    CategoricalAccuracy() {}
    CategoricalAccuracy(bool pIsBinary);
    Vec2d<T> predict(Vec2d<T> &predictions, Vec2d<T> &actualY) override;
};

// For regression networks
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
