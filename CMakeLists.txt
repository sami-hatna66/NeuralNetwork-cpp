cmake_minimum_required (VERSION 3.21.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-ffast-math")

option(BUILD_SAMPLE "Build the MNIST sample network (requires OpenCV)" ON)

option(BUILD_TESTS "Build library tests" ON)

option(BUILD_BENCH "Build library benchmarks" ON)

project(nn-cpp)

add_subdirectory(nn-cpp)

if (BUILD_SAMPLE)
    add_subdirectory(sample)
endif()

if (BUILD_TESTS)
    add_subdirectory(test)
endif()

if (BUILD_BENCH)
    add_subdirectory(bench)
endif()