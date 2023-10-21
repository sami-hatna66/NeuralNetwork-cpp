#ifndef Optimizers_hpp
#define Optimizers_hpp

#include "Layer.hpp"
#include "utils.hpp"

namespace Optimizers {

template <typename T> class OptimizerBase {
  protected:
    // Learning rate controls the amount that the weights are updated as we seek
    // to minimize the Loss
    // Small learning rates result in failure to train
    // Large learning rates make training unstable
    T learningRate;
    T currentLearningRate;
    // Learning rate decays across iterations to keep initial updates large and
    // then gradually decrease as loss falls into a local minimum
    T decay;
    T iterations;

  public:
    OptimizerBase(T pLearningRate, T PDecay);
    void setup();
    void finalize();
    T getCurrentLearningRate();

    virtual void updateParams(Layers::DenseLayer<T> &layer) = 0;
};

template <typename T>
class StochasticGradientDescent : public OptimizerBase<T> {
  private:
    // Momentum creates a rolling average of gradients over n updates
    // This average is used at each step to better inform our weight updates
    T momentum;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    StochasticGradientDescent(T pLearningRate = 1.0, T pDecay = 0.0,
                              T pMomentum = 0.0);
    void updateParams(Layers::DenseLayer<T> &layer) override;
};

template <typename T> class Adagrad : public OptimizerBase<T> {
  private:
    // Epsilon hyper-parameter prevents 0 division
    T epsilon;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    Adagrad(T pLearningRate = 1.0, T pDecay = 0.0, T pEpsilon = 0.0000001);
    void updateParams(Layers::DenseLayer<T> &layer) override;
};

template <typename T> class RMSprop : public OptimizerBase<T> {
  private:
    T epsilon;
    // Cache memory decay rate
    T rho;

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    RMSprop(T pLearningRate = 0.001, T pDecay = 0.0, T pEpsilon = 0.0000001,
            T pRho = 0.9);
    void updateParams(Layers::DenseLayer<T> &layer) override;
};

template <typename T> class Adam : public OptimizerBase<T> {
  private:
    T epsilon;
    // Decay rates for the rolling averages stored in layers weights/momentums
    T beta1; // momentum
    T beta2; // cache

    using OptimizerBase<T>::learningRate;
    using OptimizerBase<T>::currentLearningRate;
    using OptimizerBase<T>::decay;
    using OptimizerBase<T>::iterations;

  public:
    Adam(T pLearningRate = 0.001, T pDecay = 0.0, T pEpsilon = 0.0000001,
         T pBeta1 = 0.9, T pBeta2 = 0.999);
    void updateParams(Layers::DenseLayer<T> &layer) override;
};

} // namespace Optimizers

#endif
