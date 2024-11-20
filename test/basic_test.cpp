#include <gtest/gtest.h>

#include "utils.hpp"

// Check that gtest is working
TEST(HelloTest, BasicAssertions) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}