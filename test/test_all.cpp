#include <gtest/gtest.h>

#include "basic_test.cpp"
#if BUILD_UNIT_TESTS
#include "utils_unit_tests.cpp"
#include "layer_unit_tests.cpp"
#endif
#if BUILD_INTEGRATION_TESTS
#include "utils_integration_tests.cpp"
#include "layer_integration_tests.cpp"
#endif

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}