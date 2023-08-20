#include "utils.hpp"

template <typename T>
Vec2d<T> operator*(const T num, const Vec2d<T>& mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] * num;
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator*(const Vec2d<T>& mat, const T num) {
    return num * mat;
}

template <typename T>
Vec2d<T> operator*(const Vec2d<T>& a, const Vec2d<T>& b) {
    assert(a[0].size() == b.size());
    Vec2d<T> result(a.size(), std::vector<T>(b[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            for (int p = 0; p < b.size(); p++) {
                result[i][j] += a[i][p] * b[p][j];
            }
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator/(const Vec2d<T>& mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] / num;
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator/(const Vec2d<T>& a, const Vec2d<T>& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] / b[i][j];
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator-(const Vec2d<T>& mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] - num;
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator-(const Vec2d<T>& a, const Vec2d<T>& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator+(const Vec2d<T>& mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] + num;
        }
    }
    return result;
}

template <typename T>
Vec2d<T> operator+(const Vec2d<T>& a, const Vec2d<T>& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <typename T>
Vec2d<T> power(const Vec2d<T>& mat, const T exponent) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::pow(mat[i][j], exponent);
        }
    }
    return result;
}

template <typename T>
Vec2d<T> root(const Vec2d<T>& mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::sqrt(mat[i][j]);
        }
    }
    return result;
}

template <typename T>
Vec2d<T> transpose(const Vec2d<T>& mat) {
    Vec2d<T> result(mat[0].size(), std::vector<T>(mat.size(), 0.0));
    for (int i = 0; i < mat[0].size(); i++) {
        for (int j = 0; j < mat.size(); j++) {
            result[i][j] = mat[j][i];
        }
    }
    return result;
}

template <typename T>
T calculateAccuracy(const Vec2d<T>& output, const Vec2d<T>& actualY) {
    auto argmax = [](const Vec2d<T>& inp) {
        std::vector<T> out(inp.size());
        for (int i = 0; i < inp.size(); i++) {
            auto maxIter = std::max_element(inp[i].begin(), inp[i].end());
            out[i] = std::distance(inp[i].begin(), maxIter);
        }
        return out;
    };

    std::vector<T> predictions = argmax(output);
    
    std::vector<T> y;
    if (actualY.size() != 1) {
        y = argmax(actualY);
    } else {
        for (int i = 0; i < actualY[0].size(); i++) {
            y.push_back(actualY[0][i]);
        }
    }

    std::vector<bool> isPredY(y.size());
    for (int i = 0; i < y.size(); i++) {
        isPredY[i] = y[i] == predictions[i];
    }
    return std::accumulate(isPredY.begin(), isPredY.end(), 0.0) / isPredY.size();
}

// Explicit instantiations
template Vec2d<double> operator*(const double num, const Vec2d<double>& mat);
template Vec2d<float> operator*(const float num, const Vec2d<float>& mat);

template Vec2d<double> operator*(const Vec2d<double>& mat, const double num);
template Vec2d<float> operator*(const Vec2d<float>& mat, const float num);

template Vec2d<double> operator*(const Vec2d<double>& a, const Vec2d<double>& b);
template Vec2d<float> operator*(const Vec2d<float>& a, const Vec2d<float>& b);

template Vec2d<double> operator/(const Vec2d<double>& mat, const double num);
template Vec2d<float> operator/(const Vec2d<float>& mat, const float num);

template Vec2d<double> operator/(const Vec2d<double>& a, const Vec2d<double>& b);
template Vec2d<float> operator/(const Vec2d<float>& a, const Vec2d<float>& b);

template Vec2d<double> operator-(const Vec2d<double>& mat, const double num);
template Vec2d<float> operator-(const Vec2d<float>& mat, const float num);

template Vec2d<double> operator-(const Vec2d<double>& a, const Vec2d<double>& b);
template Vec2d<float> operator-(const Vec2d<float>& a, const Vec2d<float>& b);

template Vec2d<double> operator+(const Vec2d<double>& mat, const double num);
template Vec2d<float> operator+(const Vec2d<float>& mat, const float num);

template Vec2d<double> operator+(const Vec2d<double>& a, const Vec2d<double>& b);
template Vec2d<float> operator+(const Vec2d<float>& a, const Vec2d<float>& b);

template Vec2d<double> power(const Vec2d<double>& mat, const double exponent);
template Vec2d<float> power(const Vec2d<float>& mat, const float exponent);

template Vec2d<double> root(const Vec2d<double>& mat);
template Vec2d<float> root(const Vec2d<float>& mat);

template Vec2d<double> transpose(const Vec2d<double>& mat);
template Vec2d<float> transpose(const Vec2d<float>& mat);

template double calculateAccuracy(const Vec2d<double>& output, const Vec2d<double>& actualY);
template float calculateAccuracy(const Vec2d<float>& output, const Vec2d<float>& actualY);
