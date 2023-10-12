#include "Vec2d.hpp"

template <DecimalType T>
Vec2d<T>::Vec2d() {
    data = {};
    rows = 1;
    cols = 0;
}

template <DecimalType T>
Vec2d<T>::Vec2d(const std::vector<std::vector<T>>& newData) {
    rows = newData.size();
    cols = newData[0].size();
    data.reserve(rows * cols);
    for (const auto& r : newData) {
        data.insert(data.end(), r.begin(), r.end());
    }
}

template <DecimalType T>
Vec2d<T>::Vec2d(const std::vector<std::vector<T>>&& newData) {
    rows = newData.size();
    cols = newData[0].size();
    data.reserve(rows * cols);
    for (const auto& r : newData) {
        data.insert(data.end(), r.begin(), r.end());
    }
}

template <DecimalType T>
Vec2d<T>::Vec2d(const int pRows, const int pCols) {
    data.reserve(pRows * pCols);
    rows = pRows;
    cols = pCols;
}

template <DecimalType T>
Vec2d<T>::Vec2d(const int pRows, const int pCols, const T initVal) {
    data.reserve(pRows * pCols);
    std::fill(data.begin(), data.end(), initVal);
    rows = pRows;
    cols = pCols;
}

template <DecimalType T>
Vec2d<T>::Vec2d(const Vec2d<T>& other) {
    data = other.getData();
    rows = other.getRows();
    cols = other.getCols();
}

template <DecimalType T>
Vec2d<T>& Vec2d<T>::operator=(const std::vector<std::vector<T>>& newData) {
    rows = newData.size();
    cols = newData[0].size();
    data.reserve(rows * cols);
    for (const auto& r : newData) {
        data.insert(data.end(), r.begin(), r.end());
    }
    return *this;
}

template <DecimalType T>
Vec2d<T>& Vec2d<T>::operator=(const std::vector<std::vector<T>>&& newData) {
    rows = newData.size();
    cols = newData[0].size();
    data.reserve(rows * cols);
    for (const auto& r : newData) {
        data.insert(data.end(), r.begin(), r.end());
    }
    return *this;
}

template <DecimalType T>
Vec2d<T>& Vec2d<T>::operator=(const Vec2d<T>& other) {
    if (this == &other) return *this;
    data = other.getData();
    rows = other.getRows();
    cols = other.getCols();
    return *this;
}

template <DecimalType T>
Vec2d<T>& Vec2d<T>::operator=(const Vec2d<T>&& other) {
    if (this == &other) return *this;
    data = other.getData();
    rows = other.getRows();
    cols = other.getCols();
    return *this;
}

template <DecimalType T>
std::span<T> Vec2d<T>::operator[](int idx) {
    return std::span<T>(data.begin() + (idx * cols), cols);
}

template <DecimalType T>
std::vector<T> Vec2d<T>::operator[](int idx) const {
    return std::vector<T>(data.begin() + (idx * cols), data.begin() + (idx * cols) + cols);
}

template <DecimalType T>
void Vec2d<T>::push_back(std::vector<T>& newRow) {
    data.insert(data.end(), newRow.begin(), newRow.end());
}

template <DecimalType T>
void Vec2d<T>::push_back(std::vector<T>&& newRow) {
    data.insert(data.end(), newRow.begin(), newRow.end());
}

template <DecimalType T>
void Vec2d<T>::push_back(std::span<T>& newRow) {
    data.insert(data.end(), newRow.begin(), newRow.end());
}

template <DecimalType T>
void Vec2d<T>::push_back(std::span<T>&& newRow) {
    data.insert(data.end(), newRow.begin(), newRow.end());
}

template <DecimalType T>
void Vec2d<T>::clear() {
    data.clear();
}

template <DecimalType T>
void Vec2d<T>::resize(int newRows, int newCols) {
    data.clear();
    data.resize(newRows * newCols);
    rows = newRows;
    cols = newCols;
}

template <DecimalType T>
void Vec2d<T>::addToRow(int rowNum, T val) {

}

template <DecimalType T>
const std::vector<T>& Vec2d<T>::getData() const {return data;}

template <DecimalType T>
int Vec2d<T>::getRows() const {return rows;}

template <DecimalType T>
int Vec2d<T>::getCols() const {return cols;}

template <DecimalType T>
int Vec2d<T>::size() const {return rows;}

// Explicit instantiations
template class Vec2d<double>;
template class Vec2d<float>;
