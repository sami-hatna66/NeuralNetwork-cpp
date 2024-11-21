include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/a1e255a582377e1006bb88a408ac3f933ba7c916.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(basic_test basic_test.cpp)
target_link_libraries(basic_test nn-cpp GTest::gtest_main)

add_executable(utils_unit_tests utils_unit_tests.cpp)
target_link_libraries(utils_unit_tests nn-cpp GTest::gtest_main)

add_executable(utils_integration_tests utils_integration_tests.cpp)
target_link_libraries(utils_integration_tests nn-cpp GTest::gtest_main)

add_executable(test_all test_all.cpp)
target_link_libraries(test_all nn-cpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(basic_test test_all)