cmake_minimum_required (VERSION 3.21.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)

project(nn-cpp)

add_subdirectory(nn-cpp)
add_subdirectory(sample)

# cmake_minimum_required (VERSION 3.21.0)

# set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_STANDARD 20)

# project(nnfs-project)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

# add_executable(nnfs 
#     nn-cpp/utils.hpp 
#     nn-cpp/utils.cpp
#     nn-cpp/Model.hpp 
#     nn-cpp/Model.cpp
#     nn-cpp/Activations.cpp 
#     nn-cpp/Activations.hpp
#     nn-cpp/Layer.cpp
#     nn-cpp/Layer.hpp
#     nn-cpp/Loss.cpp
#     nn-cpp/Loss.hpp
#     nn-cpp/LossActivation.cpp
#     nn-cpp/LossActivation.hpp
#     nn-cpp/Optimizers.cpp
#     nn-cpp/Optimizers.hpp
#     nn-cpp/Accuracy.cpp
#     nn-cpp/Accuracy.hpp
#     nn-cpp/ModelLayer.hpp
#     nn-cpp/ModelLayer.cpp
#     nn-cpp/main.cpp 
# )

# target_link_libraries(nnfs ${OpenCV_LIBS}) 

# # find_package(Boost REQUIRED)

# # if (Boost_FOUND)
# #     message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
# #     include_directories(${Boost_INCLUDE_DIRS})
# # endif()
