#include "ModelLayer.hpp"

template <typename T>
Vec2d<T>& ModelLayer<T>::getOutput() {
    return output;
}

template <typename T>
Vec2d<T>& ModelLayer<T>::getDInputs() {
    return dInputs;
}

// Explicit instantiations
template class ModelLayer<double>;
template class ModelLayer<float>;
