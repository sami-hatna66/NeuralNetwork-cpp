#ifndef Optimizers_hpp
#define Optimizers_hpp

#include "Layer.hpp"
#include "utils.hpp"

namespace Optimizers {

template <typename T> class OptimizerBase {
  protected:
    T learningRate;
    T currentLearningRate;
    T decay;
    T iterations;

  public:
    OptimizerBase(T pLearningRate, T PDecay);
    void setup();
    void finalize();

    T getCurrentLearningRate();
};

template <typename T>
class StochasticGradientDescent : public OptimizerBase<T> {
  private:
    T momentum;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    StochasticGradientDescent(T pLearningRate = 1.0, T pDecay = 0.0,
                              T pMomentum = 0.0);
    void updateParams(Layers::DenseLayer<T> &layer);
};

template <typename T> class Adagrad : public OptimizerBase<T> {
  private:
    T epsilon;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    Adagrad(T pLearningRate = 1.0, T pDecay = 0.0, T pEpsilon = 0.0000001);
    void updateParams(Layers::DenseLayer<T> &layer);
};

template <typename T> class RMSprop : public OptimizerBase<T> {
  private:
    T epsilon;
    T rho;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    RMSprop(T pLearningRate = 0.001, T pDecay = 0.0, T pEpsilon = 0.0000001,
            T pRho = 0.9);
    void updateParams(Layers::DenseLayer<T> &layer);
};

template <typename T> class Adam : public OptimizerBase<T> {
  private:
    T epsilon;
    T beta1;
    T beta2;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    Adam(T pLearningRate = 0.001, T pDecay = 0.0, T pEpsilon = 0.0000001,
         T pBeta1 = 0.9, T pBeta2 = 0.999);
    void updateParams(Layers::DenseLayer<T> &layer);
};

} // namespace Optimizers

#endif
