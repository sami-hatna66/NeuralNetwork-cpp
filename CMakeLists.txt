cmake_minimum_required (VERSION 3.21.0)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)

project(nnfs-project)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

add_executable(nnfs 
    src/utils.hpp 
    src/utils.cpp
    src/Vec2d.hpp
    src/Vec2d.cpp
    src/Model.hpp 
    src/Model.cpp
    src/Activations.cpp 
    src/Activations.hpp
    src/Layer.cpp
    src/Layer.hpp
    src/Loss.cpp
    src/Loss.hpp
    src/LossActivation.cpp
    src/LossActivation.hpp
    src/Optimizers.cpp
    src/Optimizers.hpp
    src/Accuracy.cpp
    src/Accuracy.hpp
    src/ModelLayer.hpp
    src/ModelLayer.cpp
    src/main.cpp 
)

# find_package(Boost REQUIRED)

# if (Boost_FOUND)
#     message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
#     include_directories(${Boost_INCLUDE_DIRS})
# endif()
