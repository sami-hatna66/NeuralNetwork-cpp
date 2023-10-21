cmake_minimum_required (VERSION 3.21.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)

option(BUILD_SAMPLE "Build the MNIST sample network (requires OpenCV)" ON)

project(nn-cpp)

add_subdirectory(nn-cpp)

if (BUILD_SAMPLE)
    add_subdirectory(sample)
endif()
