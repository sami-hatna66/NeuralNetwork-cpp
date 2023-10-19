#ifndef Model_hpp
#define Model_hpp

#include "Accuracy.hpp"
#include "Activations.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "LossActivation.hpp"
#include "ModelLayer.hpp"
#include "Optimizers.hpp"
#include "utils.hpp"

#include <vector>

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
class Model {
  private:
    std::vector<std::shared_ptr<ModelLayer<T>>> layers;
    LossType<T> loss;
    AccuracyType<T> accuracy;
    OptimizerType<T> optimizer;
    LossActivation::SoftmaxCCE<T> lossAcc;
    bool useLossAcc = false;

  public:
    Model();
    template <typename Derived> void addLayer(std::shared_ptr<Derived> layer);
    void setOptimizer(OptimizerType<T> &pOptimizer);
    void setAccuracy(AccuracyType<T> &pAccuracy);
    void setLoss(LossType<T> &pLoss);
    void prepare();
    void train(std::unique_ptr<Vec2d<T>> X, std::unique_ptr<Vec2d<T>>y,
        std::unique_ptr<Vec2d<T>> testX, std::unique_ptr<Vec2d<T>> testY, 
        int epochs = 1, int batchSize = 0);
    Vec2d<T> compute(Vec2d<T> &X, LayerMode mode);
    void backward(Vec2d<T> &output, Vec2d<T> &actualY);
    Vec2d<T> predict(Vec2d<T>& inp, int batchSize = 0);
};

#endif
