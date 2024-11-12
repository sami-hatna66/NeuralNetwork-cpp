#include <gtest/gtest.h>

#include "utils.hpp"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

typedef ::testing::Types<float, double> NumericTypes;

template <typename T>
class AbsTest : public ::testing::Test {};

TYPED_TEST_SUITE(AbsTest, NumericTypes);

TYPED_TEST(AbsTest, AbsTestSmall) {
    Vec2d<TypeParam> input {{4, 9}};
    auto result = abs(input);

    Vec2d<TypeParam> expectedResult {{4, 9}};

    EXPECT_EQ(result[0][0], expectedResult[0][0]);
    EXPECT_EQ(result[0][1], expectedResult[0][1]);
}