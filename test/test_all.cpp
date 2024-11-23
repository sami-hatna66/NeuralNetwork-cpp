#include <gtest/gtest.h>

#include "basic_test.cpp"
#include "utils_unit_tests.cpp"
#include "utils_integration_tests.cpp"
#include "layer_unit_tests.cpp"
#include "layer_integration_tests.cpp"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}