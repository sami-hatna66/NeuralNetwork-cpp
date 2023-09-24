#ifndef Model_hpp
#define Model_hpp

#include "Accuracy.hpp"
#include "Activations.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "LossActivation.hpp"
#include "Optimizers.hpp"
#include "utils.hpp"

#include <any>
#include <vector>

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
class Model {
  private:
    std::vector<std::any> layers;
    std::vector<std::any> trainableLayers;
    Vec2d<T> SoftmaxClassifierOutput;
    LossType<T> loss;
    AccuracyType<T> accuracy;
    OptimizerType<T> optimizer;
    LossActivation::SoftmaxCCE<T> lossacc;
    Layers::InputLayer<T> inputLayer;

  public:
    Model();
    void addLayer(std::any layer);
    void setOptimizer(OptimizerType<T> pOptimizer);
    void setAccuracy(AccuracyType<T> pAccuracy);
    void setLoss(LossType<T> pLoss);
    void prepare();
    void train(Vec2d<T> X, Vec2d<T> y, int epochs = 1);
    Vec2d<T> compute(Vec2d<T> X, LayerMode mode);
    void backward(Vec2d<T> output, Vec2d<T> actualY);
};

#endif
