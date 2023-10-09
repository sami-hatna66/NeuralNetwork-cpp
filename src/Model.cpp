#include "Model.hpp"

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
Model<T, AccuracyType, OptimizerType, LossType>::Model() {}

template <typename T, template <typename> class AccuracyType, 
          template <typename> class OptimizerType, template <typename> class LossType>
template <typename Derived>
void Model<T, AccuracyType, OptimizerType, LossType>::addLayer(std::shared_ptr<Derived> layer) {
    layers.push_back(std::static_pointer_cast<ModelLayer<T>>(layer));
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

// template <typename T, template <typename> class AccuracyType, 
//           template <typename> class OptimizerType, template <typename> class LossType>
// void Model<T, AccuracyType, OptimizerType, LossType>::prepare() {
//     Layers::InputLayer<T> inputLayer;
//     layers.insert(layers.begin(), {inputLayer, false});

//     Layers::InputLayer<T> outputLayer;
//     layers.insert(layers.end(), {outputLayer, false});

//     for (auto& layer : layers) {
//         if (layer.first.type() == typeid(Layers::DenseLayer<T>)) {
//             layer.second = true;
//         }
//     }

//     if (layers.back().first.type() == typeid(Activations::Softmax<T>) &&
//         std::is_same_v<LossType<T>, Loss::CategoricalCrossEntropy<T>>) {
//         useLossAcc = true;
//     }
// }

// template <typename T, template <typename> class AccuracyType, 
//           template <typename> class OptimizerType, template <typename> class LossType>
// void Model<T, AccuracyType, OptimizerType, LossType>::train(Vec2d<T> X, Vec2d<T> y, int epochs) {

// }



// template <typename T, template <typename> class AccuracyType, 
//           template <typename> class OptimizerType, template <typename> class LossType>
// Vec2d<T> Model<T, AccuracyType, OptimizerType, LossType>::compute(Vec2d<T> X, LayerMode mode) {
//     std::any_cast<Layers::InputLayer<T>>(layers.front().first).compute(X, mode);
//     for (int i = 1; i < layers.size(); i++) {
//         if (layers[i].first.type() == typeid(Layers::DenseLayer<T>)) {
//             std::any_cast
//         }
//     }

//     // return final layer output
//     return std::any_cast<Layers::InputLayer<T>>(layers.back().first).getOutput();
// }

// template <typename T, template <typename> class AccuracyType, 
//           template <typename> class OptimizerType, template <typename> class LossType>
// void Model<T, AccuracyType, OptimizerType, LossType>::backward(Vec2d<T> output, Vec2d<T> actualY) {

// }


// explicit instantiations
#define INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, LOSS)\
    template class Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>;\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Layers::DenseLayer<DTYPE>>(std::shared_ptr<Layers::DenseLayer<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Layers::DropoutLayer<DTYPE>>(std::shared_ptr<Layers::DropoutLayer<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Layers::InputLayer<DTYPE>>(std::shared_ptr<Layers::InputLayer<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Activations::Linear<DTYPE>>(std::shared_ptr<Activations::Linear<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Activations::Sigmoid<DTYPE>>(std::shared_ptr<Activations::Sigmoid<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Activations::Softmax<DTYPE>>(std::shared_ptr<Activations::Softmax<DTYPE>> layer);\
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<Activations::Relu<DTYPE>>(std::shared_ptr<Activations::Relu<DTYPE>> layer);

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
