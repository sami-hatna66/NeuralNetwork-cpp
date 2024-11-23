#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Activations.hpp"
#include "utils.hpp"

typedef ::testing::Types<float, double> TestTypes;

template <typename T>
class LayerIntegrationTests : public ::testing::Test {};

TYPED_TEST_SUITE(LayerIntegrationTests, TestTypes);

TYPED_TEST(LayerIntegrationTests, DenseReluDenseCCE) {
    Vec2d<TypeParam> weights;
    populateVec2dWithSequentialData(weights, 3, 3);

    Vec2d<TypeParam> biases;
    populateVec2dWithSequentialData(biases, 1, 3);

    Layers::DenseLayer<TypeParam> dense1 {weights, biases};
    Activations::Relu<TypeParam> relu {};
    Layers::DenseLayer<TypeParam> dense2 {weights, biases};
    Loss::CategoricalCrossEntropy<TypeParam> cce {};

    Vec2d<TypeParam> inputs;
    populateVec2dWithSequentialData(inputs, 3, 3);

    Vec2d<TypeParam> actualY;
    populateVec2dWithSequentialData(actualY, 3, 3);

    dense1.compute(inputs);
    relu.compute(dense1.getOutput());
    dense2.compute(relu.getOutput());
    auto result = cce.calculate(dense2.getOutput(), actualY);

    EXPECT_NEAR(result, -2.209347, 0.1);
}

TYPED_TEST(LayerIntegrationTests, DenseSoftmaxDenseBCE) {
    Vec2d<TypeParam> weights;
    populateVec2dWithSequentialData(weights, 3, 3);

    Vec2d<TypeParam> biases;
    populateVec2dWithSequentialData(biases, 1, 3);

    Layers::DenseLayer<TypeParam> dense1 {weights, biases};
    Activations::Softmax<TypeParam> softmax {};
    Layers::DenseLayer<TypeParam> dense2 {weights, biases};
    Loss::BinaryCrossEntropy<TypeParam> bce {};

    Vec2d<TypeParam> inputs;
    populateVec2dWithSequentialData(inputs, 3, 3, 0, 1);

    Vec2d<TypeParam> actualY;
    populateVec2dWithSequentialData(actualY, 3, 3);

    dense1.compute(inputs);
    softmax.compute(dense1.getOutput());
    dense2.compute(softmax.getOutput());
    auto result = bce.calculate(dense2.getOutput(), actualY);

    EXPECT_NEAR(result, -48, 1);
}

TYPED_TEST(LayerIntegrationTests, DenseSigmoidDenseMSE) {
    Vec2d<TypeParam> weights;
    populateVec2dWithSequentialData(weights, 3, 3);

    Vec2d<TypeParam> biases;
    populateVec2dWithSequentialData(biases, 1, 3);

    Layers::DenseLayer<TypeParam> dense1 {weights, biases};
    Activations::Sigmoid<TypeParam> sigmoid {};
    Layers::DenseLayer<TypeParam> dense2 {weights, biases};
    Loss::MeanSquaredError<TypeParam> mse {};

    Vec2d<TypeParam> inputs;
    populateVec2dWithSequentialData(inputs, 3, 3);

    Vec2d<TypeParam> actualY;
    populateVec2dWithSequentialData(actualY, 3, 3);

    dense1.compute(inputs);
    sigmoid.compute(dense1.getOutput());
    dense2.compute(sigmoid.getOutput());
    auto result = mse.calculate(dense2.getOutput(), actualY);

    EXPECT_NEAR(result, 93, 1);
}

TYPED_TEST(LayerIntegrationTests, DenseLinearDenseMAE) {
    Vec2d<TypeParam> weights;
    populateVec2dWithSequentialData(weights, 3, 3);

    Vec2d<TypeParam> biases;
    populateVec2dWithSequentialData(biases, 1, 3);

    Layers::DenseLayer<TypeParam> dense1 {weights, biases};
    Activations::Linear<TypeParam> linear {};
    Layers::DenseLayer<TypeParam> dense2 {weights, biases};
    Loss::MeanAbsoluteError<TypeParam> mae {};

    Vec2d<TypeParam> inputs;
    populateVec2dWithSequentialData(inputs, 3, 3);

    Vec2d<TypeParam> actualY;
    populateVec2dWithSequentialData(actualY, 3, 3);

    dense1.compute(inputs);
    linear.compute(dense1.getOutput());
    dense2.compute(linear.getOutput());
    auto result = mae.calculate(dense2.getOutput(), actualY);

    EXPECT_NEAR(result, 735, 1);
}
