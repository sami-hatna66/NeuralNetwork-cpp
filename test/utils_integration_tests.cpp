#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "utils.hpp"

typedef ::testing::Types<float, double> TestTypes;

template <typename T>
class UtilsIntegrationTests : public ::testing::Test {};

TYPED_TEST_SUITE(UtilsIntegrationTests, TestTypes);

TYPED_TEST(UtilsIntegrationTests, MulMulPow) {
    Vec2d<TypeParam> a {{4, 9}, {3, 4}};
    Vec2d<TypeParam> b {{4, 9, 5}, {3, 4, 6}};
    Vec2d<TypeParam> c {{4, 9}, {3, 4}, {5, 6}};

    auto result = power((a * b) * c, (TypeParam)2.0);

    Vec2d<TypeParam> expected {{574564, 1252161}, {176400, 386884}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsIntegrationTests, MulDivMean) {
    Vec2d<TypeParam> a {{2,4}, {6,8}, {10,15}};
    Vec2d<TypeParam> b {{2,2}, {2,2}};
    Vec2d<TypeParam> c {{2,2}, {2,2}, {2,2}};

    auto result = mean((a * b) / c);

    TypeParam expected = 15;

    EXPECT_EQ(result, expected);
}

TYPED_TEST(UtilsIntegrationTests, MulAddRoot) {
    Vec2d<TypeParam> a {{2,4}, {6,8}, {10,15}};
    Vec2d<TypeParam> b {{2,2}, {2,2}};
    Vec2d<TypeParam> c {{4,4}, {8,8}, {14,14}};

    auto result = root((a * b) + c);

    Vec2d<TypeParam> expected {{4,4},{6,6},{8,8}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsIntegrationTests, MulSubExp) {
    Vec2d<TypeParam> a {{3,9}, {1,8}, {3,21}};
    Vec2d<TypeParam> b {{2,1}, {2,4}};
    Vec2d<TypeParam> c {{23,37}, {15,29}, {43,81}};

    auto result = exp((a * b) - c);

    Vec2d<TypeParam> expected {{2.718281746,7.389056206},{20.08553696,54.59814835},{148.4131622,403.4288025}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsIntegrationTests, DivAddLog) {
    Vec2d<TypeParam> a {{1,2}, {3,4}};
    Vec2d<TypeParam> b {{5,6}, {7,8}};
    Vec2d<TypeParam> c {{9,10}, {11,12}};

    auto result = log((a / b) + c);

    Vec2d<TypeParam> expected {{2.219203472,2.335374832},{2.436116457,2.525728703}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsIntegrationTests, DivSubAbs) {
    Vec2d<TypeParam> a {{1,2}, {3,4}};
    Vec2d<TypeParam> b {{5,6}, {7,8}};
    Vec2d<TypeParam> c {{9,10}, {11,12}};

    auto result = abs((a / b) - c);

    Vec2d<TypeParam> expected {{8.8,9.666666667},{10.57142857,11.5}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsIntegrationTests, AddSubTranspose) {
    Vec2d<TypeParam> a {{1,2}, {3,4}, {1,1}};
    Vec2d<TypeParam> b {{5,6}, {7,8}, {1,1}};
    Vec2d<TypeParam> c {{9,10}, {11,12}, {1,1}};

    auto result = transpose((a / b) - c);

    Vec2d<TypeParam> expected {{-8.8,-10.57142857,0}, {-9.666666667,-11.5,0}};

    VEC2D_EXPECT_EQ(result, expected)
}


































