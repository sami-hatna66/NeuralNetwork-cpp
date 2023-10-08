#include "Model.hpp"

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
Model<T, AccuracyType, OptimizerType, LossType>::Model() {}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::addLayer(std::any layer) {
    layers.push_back(layer);
}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setOptimizer(OptimizerType<T> pOptimizer) {
    optimizer = pOptimizer;
}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setAccuracy(AccuracyType<T> pAccuracy) {
    accuracy = pAccuracy;
}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setLoss(LossType<T> pLoss) {
    loss = pLoss;
}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::prepare() {
    inputLayer = Layers::InputLayer<T>();

    // find all layers with attribute weights and add to trainableLayers
    // loss remember trainable layers
    // if output activation is softmax and loss is CCE create activationsoftmaxcce
}

// explicit instantiations
#define INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, LOSS)\
    template class Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>;

#define INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, OPTIMIZER)\
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::BinaryCrossEntropy)\
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::CategoricalCrossEntropy)\
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanAbsoluteError)\
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanSquaredError)

#define INST_MODEL_WITH_OPTIMIZER(DTYPE, ACCURACY)\
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::Adagrad)\
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::Adam)\
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::RMSprop)\
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::StochasticGradientDescent)

#define INST_MODEL_WITH_ACCURACY(DTYPE)\
    INST_MODEL_WITH_OPTIMIZER(DTYPE, Accuracy::CategoricalAccuracy)\
    INST_MODEL_WITH_OPTIMIZER(DTYPE, Accuracy::RegressionAccuracy)

#define INST_MODEL_WITH_TYPE() \
    INST_MODEL_WITH_ACCURACY(float)\
    INST_MODEL_WITH_ACCURACY(double)

INST_MODEL_WITH_TYPE()
