#include "Model.hpp"

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
Model<T, AccuracyType, OptimizerType, LossType>::Model() {}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
template <typename Derived>
void Model<T, AccuracyType, OptimizerType, LossType>::addLayer(
    std::shared_ptr<Derived> layer) {
    layers.push_back(std::static_pointer_cast<ModelLayer<T>>(layer));
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setOptimizer(
    OptimizerType<T> &pOptimizer) {
    optimizer = pOptimizer;
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setAccuracy(
    AccuracyType<T> &pAccuracy) {
    accuracy = pAccuracy;
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::setLoss(
    LossType<T> &pLoss) {
    loss = pLoss;
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::prepare() {
    auto inputLayer = std::make_shared<Layers::InputLayer<T>>();
    layers.insert(layers.begin(),
                  std::static_pointer_cast<ModelLayer<T>>(inputLayer));

    if (std::dynamic_pointer_cast<Activations::Softmax<T>>(layers.back()) &&
        std::is_same_v<LossType<T>, Loss::CategoricalCrossEntropy<T>>) {
        useLossAcc = true;
    }
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::train(Vec2d<T> &X,
                                                            Vec2d<T> &y,
                                                            int epochs) {
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; i++) {
        auto output = compute(X, LayerMode::Training);

        auto dataLoss = loss.calculate(output, y);
        T regLoss = 0;
        for (auto layer : layers) {
            if (layer->getIsTrainable()) {
                regLoss += loss.calculateRegLoss(
                    *std::static_pointer_cast<Layers::DenseLayer<T>>(layer));
            }
        }
        auto lossVal = dataLoss + regLoss;

        auto predictions = layers[layers.size() - 1]->predict(output);
        auto calcAccuracy = accuracy.calculate(predictions, y);

        backward(output, y);

        optimizer.setup();
        for (auto layer : layers) {
            if (layer->getIsTrainable()) {
                optimizer.updateParams(
                    *std::static_pointer_cast<Layers::DenseLayer<T>>(layer));
            }
        }
        optimizer.finalize();

        if (i % 100 == 0) {
            auto endTime = std::chrono::high_resolution_clock::now();
            std::cout << "epoch: " << i << std::fixed << std::setprecision(3)
                      << ", accuracy: " << calcAccuracy << std::fixed
                      << std::fixed << std::setprecision(3)
                      << ", loss: " << lossVal << " (data loss: " << dataLoss
                      << ", reg loss: " << regLoss << ")" << std::fixed
                      << std::setprecision(3)
                      << ", lr: " << optimizer.getCurrentLearningRate()
                      << ", time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             endTime - startTime)
                             .count()
                      << " ms" << std::endl;
            startTime = std::chrono::high_resolution_clock::now();
        }
    }
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
Vec2d<T>
Model<T, AccuracyType, OptimizerType, LossType>::compute(Vec2d<T> &X,
                                                         LayerMode mode) {
    layers[0]->compute(X, mode);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->compute(layers[i - 1]->getOutput(), mode);
    }

    return layers.back()->getOutput();
}

template <typename T, template <typename> class AccuracyType,
          template <typename> class OptimizerType,
          template <typename> class LossType>
void Model<T, AccuracyType, OptimizerType, LossType>::backward(
    Vec2d<T> &output, Vec2d<T> &actualY) {
    if (useLossAcc) {
        lossAcc.backward(output, actualY);
        if (layers.size() > 1) {
            layers[layers.size() - 1]->setDInputs(lossAcc.getDInputs());
            for (int i = layers.size() - 2; i >= 0; i--) {
                layers[i]->backward(layers[i + 1]->getDInputs());
            }
        }
    } else {
        loss.backward(output, actualY);
        layers[layers.size() - 1]->backward(loss.getDInputs());
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers[i]->backward(layers[i + 1]->getDInputs());
        }
    }
}

// explicit instantiations
#define INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, LOSS)                           \
    template class Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>;                    \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Layers::DenseLayer<DTYPE>>(                                            \
        std::shared_ptr<Layers::DenseLayer<DTYPE>> layer);                     \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Layers::DropoutLayer<DTYPE>>(                                          \
        std::shared_ptr<Layers::DropoutLayer<DTYPE>> layer);                   \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Layers::InputLayer<DTYPE>>(                                            \
        std::shared_ptr<Layers::InputLayer<DTYPE>> layer);                     \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Activations::Linear<DTYPE>>(                                           \
        std::shared_ptr<Activations::Linear<DTYPE>> layer);                    \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Activations::Sigmoid<DTYPE>>(                                          \
        std::shared_ptr<Activations::Sigmoid<DTYPE>> layer);                   \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Activations::Softmax<DTYPE>>(                                          \
        std::shared_ptr<Activations::Softmax<DTYPE>> layer);                   \
    template void Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>::addLayer<           \
        Activations::Relu<DTYPE>>(                                             \
        std::shared_ptr<Activations::Relu<DTYPE>> layer);

#define INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, OPTIMIZER)                       \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::BinaryCrossEntropy)           \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::CategoricalCrossEntropy)      \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanAbsoluteError)            \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanSquaredError)

#define INST_MODEL_WITH_OPTIMIZER(DTYPE, ACCURACY)                             \
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::Adagrad)                 \
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::Adam)                    \
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::RMSprop)                 \
    INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, Optimizers::StochasticGradientDescent)

#define INST_MODEL_WITH_ACCURACY(DTYPE)                                        \
    INST_MODEL_WITH_OPTIMIZER(DTYPE, Accuracy::CategoricalAccuracy)            \
    INST_MODEL_WITH_OPTIMIZER(DTYPE, Accuracy::RegressionAccuracy)

#define INST_MODEL_WITH_TYPE()                                                 \
    INST_MODEL_WITH_ACCURACY(float)                                            \
    INST_MODEL_WITH_ACCURACY(double)

INST_MODEL_WITH_TYPE()
