#include "Model.hpp"

// explicit instantiations
#define INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, LOSS)                           \
    template class Model<DTYPE, ACCURACY, OPTIMIZER, LOSS>;

#define INST_MODEL_WITH_LOSS(DTYPE, ACCURACY, OPTIMIZER)                       \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::BinaryCrossEntropy)           \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::CategoricalCrossEntropy)      \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanAbsoluteError)            \
    INST_MODEL(DTYPE, ACCURACY, OPTIMIZER, Loss::MeanSquaredError)
// float, categorical, adam, binary crossentropy
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
