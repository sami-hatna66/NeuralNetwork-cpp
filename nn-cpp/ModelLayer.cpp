#include "ModelLayer.hpp"

template <typename T> Vec2d<T> &ModelLayer<T>::getOutput() { return output; }

template <typename T> Vec2d<T> &ModelLayer<T>::getDInputs() { return dInputs; }

template <typename T> void ModelLayer<T>::setDInputs(Vec2d<T> &newDInputs) {
    dInputs = newDInputs;
}

template <typename T> bool ModelLayer<T>::getIsTrainable() {
    return isTrainable;
}

template <typename T> Vec2d<T> ModelLayer<T>::predict(Vec2d<T> &outputs) {
    throw "Tried to call predict on a layer which wasn't an activation layer!";
}

// Explicit instantiations
template class ModelLayer<double>;
template class ModelLayer<float>;
