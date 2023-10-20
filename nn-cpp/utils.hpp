#ifndef utils_hpp
#define utils_hpp

#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

template <typename T>
concept DecimalType = std::is_floating_point_v<T>;

template <DecimalType T> using Vec2d = std::vector<std::vector<T>>;

enum class LayerMode { Training, Eval };

template <typename T> static void printVec(std::vector<T> &inp) {
    std::cout << std::setprecision(10);
    for (auto val : inp) {
        std::cout << val << ", ";
    }
    std::cout << std::endl << std::endl;
}

template <typename T> static void printVec(Vec2d<T> &inp) {
    std::cout << std::setprecision(10);
    for (auto& row : inp) {
        for (auto val : row) {
            std::cout << val << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

template <typename T> static void printVec(const Vec2d<T> &inp) {
    std::cout << std::setprecision(10);
    for (auto& row : inp) {
        for (auto val : row) {
            std::cout << val << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

template <typename T> Vec2d<T> operator*(const T num, const Vec2d<T> &mat);

template <typename T> Vec2d<T> operator*(const Vec2d<T> &mat, const T num);

template <typename T> Vec2d<T> operator*(const Vec2d<T> &a, const Vec2d<T> &b);

template <typename T> Vec2d<T> operator/(const Vec2d<T> &mat, const T num);

template <typename T> Vec2d<T> operator/(const T num, const Vec2d<T> &mat);

template <typename T> Vec2d<T> operator/(const Vec2d<T> &a, const Vec2d<T> &b);

template <typename T> Vec2d<T> operator-(const Vec2d<T> &mat, const T num);

template <typename T> Vec2d<T> operator-(const T num, const Vec2d<T> &mat);

template <typename T> Vec2d<T> operator-(const Vec2d<T> &a, const Vec2d<T> &b);

template <typename T> Vec2d<T> operator+(const Vec2d<T> &mat, const T num);

template <typename T> Vec2d<T> operator+(const Vec2d<T> &a, const Vec2d<T> &b);

template <typename T> Vec2d<T> power(const Vec2d<T> &mat, const T exponent);

template <typename T> Vec2d<T> root(const Vec2d<T> &mat);

template <typename T> Vec2d<T> exp(const Vec2d<T> &mat);

template <typename T> Vec2d<T> log(const Vec2d<T> &mat);

template <typename T> Vec2d<T> abs(const Vec2d<T> &mat);

template <typename T> Vec2d<T> transpose(const Vec2d<T> &mat);

template <typename T> T mean(const Vec2d<T> &mat);

template <typename T>
Vec2d<T> eltwiseMult(const Vec2d<T> &a, const Vec2d<T> &b);

template <typename T>
T calculateAccuracy(const Vec2d<T> &output, const Vec2d<T> &actualY);

#endif
