include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/a1e255a582377e1006bb88a408ac3f933ba7c916.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

if (ENABLE_GCOV)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-g -O0 --coverage)
    add_link_options(--coverage)
  else()
    message(FATAL_ERROR "Code coverage is only supported with GCC or Clang.")
  endif()
endif()

add_executable(basic_test basic_test.cpp)
target_link_libraries(basic_test nn-cpp GTest::gtest_main)

if (BUILD_UNIT_TESTS)
  add_executable(utils_unit_tests utils_unit_tests.cpp)
  target_link_libraries(utils_unit_tests nn-cpp GTest::gtest_main)

  add_executable(layer_unit_tests layer_unit_tests.cpp)
  target_link_libraries(layer_unit_tests nn-cpp GTest::gtest_main)
endif()

if (BUILD_INTEGRATION_TESTS)
  add_executable(utils_integration_tests utils_integration_tests.cpp)
  target_link_libraries(utils_integration_tests nn-cpp GTest::gtest_main)

  add_executable(layer_integration_tests layer_integration_tests.cpp)
  target_link_libraries(layer_integration_tests nn-cpp GTest::gtest_main)
endif()

if (BUILD_SYSTEM_TESTS)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_executable(system_tests system_tests.cpp)
  target_link_libraries(system_tests nn-cpp GTest::gtest_main ${OpenCV_LIBS})
endif()

add_executable(test_all test_all.cpp)
target_compile_definitions(test_all PRIVATE BUILD_UNIT_TESTS=$<BOOL:${BUILD_UNIT_TESTS}> BUILD_INTEGRATION_TESTS=$<BOOL:${BUILD_INTEGRATION_TESTS}> BUILD_SYSTEM_TESTS=$<BOOL:${BUILD_SYSTEM_TESTS}>)
target_link_libraries(test_all nn-cpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(basic_test test_all)

if (ENABLE_GCOV)
  find_program(GCOVR_EXECUTABLE gcovr REQUIRED)

  set(COVERAGE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/coverage)
  file(MAKE_DIRECTORY ${COVERAGE_OUTPUT_DIR})

  add_custom_target(coverage
    COMMAND ${GCOVR_EXECUTABLE}
            --root ${CMAKE_SOURCE_DIR}
            --exclude ${CMAKE_BINARY_DIR}
            --exclude "${CMAKE_SOURCE_DIR}.*\\.hpp$"
            --xml ${COVERAGE_OUTPUT_DIR}/coverage.xml
            --html ${COVERAGE_OUTPUT_DIR}/coverage.html
            --html-details
    COMMENT "Generating code coverage report in ${COVERAGE_OUTPUT_DIR}"
  )
endif()
