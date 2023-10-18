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
void Model<T, AccuracyType, OptimizerType, LossType>::train(std::unique_ptr<Vec2d<T>> X,
                                                            std::unique_ptr<Vec2d<T>> y,
                                                            std::unique_ptr<Vec2d<T>> testX,
                                                            std::unique_ptr<Vec2d<T>> testY,
                                                            int epochs,
                                                            int batchSize) {
    auto startTime = std::chrono::high_resolution_clock::now();

    int numSteps = 1;
    if (batchSize != 0) {
        // Ceiling divide
        numSteps = (X->size() + batchSize - 1) / batchSize;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;

        loss.newPass();
        accuracy.newPass();

        for (int step = 0; step < numSteps; step++) {
            Vec2d<T> batchX;
            Vec2d<T> batchY;
            if (batchSize == 0) {
                batchX = Vec2d<T>(*X);
                batchY = Vec2d<T>(*y);
            } else {
                int startRow = step * batchSize;
                int endRow = (step + 1) * batchSize;
                if (endRow > X->size()) endRow = X->size();
                for (int i = startRow; i < endRow; i++) {
                    batchX.push_back(std::vector<T>((*X)[i].begin(), (*X)[i].end()));
                }
                batchY.push_back(std::vector<T>((*y)[0].begin() + startRow,
                                                (*y)[0].begin() + endRow));
            }

            auto output = compute(batchX, LayerMode::Training);

            auto dataLoss = loss.calculate(output, batchY);
            T regLoss = 0;
            for (auto layer : layers) {
                if (layer->getIsTrainable()) {
                    regLoss += loss.calculateRegLoss(
                        *std::static_pointer_cast<Layers::DenseLayer<T>>(
                            layer));
                }
            }
            auto lossVal = dataLoss + regLoss;

            auto predictions = layers[layers.size() - 1]->predict(output);
            auto calcAccuracy = accuracy.calculate(predictions, batchY);

            backward(output, batchY);

            optimizer.setup();
            for (auto layer : layers) {
                if (layer->getIsTrainable()) {
                    optimizer.updateParams(
                        *std::static_pointer_cast<Layers::DenseLayer<T>>(
                            layer));
                }
            }
            optimizer.finalize();

            if (step % 100 == 0 || step == numSteps - 1) {
                auto endTime = std::chrono::high_resolution_clock::now();
                std::cout
                    << "step: " << step << std::fixed << std::setprecision(3)
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

        auto epochLoss = loss.calculateAccumulatedLoss();
        auto epochAccuracy = accuracy.calculateAccumulatedAcc();

        std::cout << "training, acc: " << epochAccuracy << ", loss: " << epochLoss << std::endl;

        if (testX != nullptr && testY != nullptr) {
            loss.newPass();
            accuracy.newPass();

            int valSteps = 1;
            if (batchSize != 0) {
                // Ceiling divide
                valSteps = (testX->size() + batchSize - 1) / batchSize;
            }

            for (int step = 0; step < valSteps; step++) {
                Vec2d<T> batchTestX;
                Vec2d<T> batchTestY;
                if (batchSize == 0) {
                    batchTestX = Vec2d<T>(*testX);
                    batchTestY = Vec2d<T>(*testY);
                } else {
                    int startRow = step * batchSize;
                    int endRow = (step + 1) * batchSize;
                    if (endRow > testX->size()) endRow = testX->size();
                    for (int i = startRow; i < endRow; i++) {
                        batchTestX.push_back(std::vector<T>((*testX)[i].begin(), (*testX)[i].end()));
                    }
                    batchTestY.push_back(std::vector<T>((*testY)[0].begin() + startRow,
                                                        (*testY)[0].begin() + endRow));
                }

                auto output = compute(batchTestX, LayerMode::Eval);

                auto predictions = layers[layers.size() - 1]->predict(output);
                auto calcAccuracy = accuracy.calculate(predictions, batchTestY);
            }

            auto valAccuracy = accuracy.calculateAccumulatedAcc();

            std::cout << "validation, acc: " << valAccuracy << std::endl;
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
