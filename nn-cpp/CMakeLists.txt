add_library(nn-cpp
    utils.hpp 
    utils.cpp
    Model.hpp 
    Model.cpp
    Activations.cpp 
    Activations.hpp
    Layer.cpp
    Layer.hpp
    Loss.cpp
    Loss.hpp
    LossActivation.cpp
    LossActivation.hpp
    Optimizers.cpp
    Optimizers.hpp
    Accuracy.cpp
    Accuracy.hpp
    ModelLayer.hpp
    ModelLayer.cpp
)

target_include_directories(nn-cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
