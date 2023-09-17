#include "utils.hpp"

const int NUMTHREADS = std::thread::hardware_concurrency();

template <typename T>
void matmulKernel(const Vec2d<T> &a, const Vec2d<T> &b, Vec2d<T> &result,
                    const int startRow, const int endRow) {
    const int colsA = a[0].size();
    const int colsB = b[0].size();
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

template <typename T> Vec2d<T> operator*(const Vec2d<T> &a, const Vec2d<T> &b) {
    assert(a[0].size() == b.size());
    Vec2d<T> result(a.size(), std::vector<T>(b[0].size(), 0.0));

    int rowsPerThread = (a.size() + NUMTHREADS - 1) / NUMTHREADS;

    std::vector<std::thread> threads;
    threads.reserve(NUMTHREADS);

    for (int i = 0; i < NUMTHREADS; i++) {
        const int startRow = i * rowsPerThread;
        const int endRow = std::min((i + 1) * rowsPerThread, (int)a.size());
        threads.push_back(std::thread(matmulKernel<T>, std::cref(a),
                                      std::cref(b), std::ref(result), startRow,
                                      endRow));
    }

    for (int i = 0; i < NUMTHREADS; i++) {
        threads[i].join();
    }

    return result;
}

template <typename T> Vec2d<T> operator*(const T num, const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] * num;
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator*(const Vec2d<T> &mat, const T num) {
    return num * mat;
}

template <typename T> Vec2d<T> operator/(const Vec2d<T> &mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] / num;
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator/(const T num, const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = num / mat[i][j];
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator/(const Vec2d<T> &a, const Vec2d<T> &b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] / b[i][j];
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator-(const Vec2d<T> &mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] - num;
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator-(const T num, const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = num - mat[i][j];
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator-(const Vec2d<T> &a, const Vec2d<T> &b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator+(const Vec2d<T> &mat, const T num) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] + num;
        }
    }
    return result;
}

template <typename T> Vec2d<T> operator+(const Vec2d<T> &a, const Vec2d<T> &b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <typename T> Vec2d<T> power(const Vec2d<T> &mat, const T exponent) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::pow(mat[i][j], exponent);
        }
    }
    return result;
}

template <typename T> Vec2d<T> root(const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            assert(mat[i][j] >= 0);
            result[i][j] = std::sqrt(mat[i][j]);
        }
    }
    return result;
}

template <typename T> Vec2d<T> exp(const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::exp(mat[i][j]);
        }
    }
    return result;
}

template <typename T> Vec2d<T> log(const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            assert(mat[i][j] != 0);
            result[i][j] = std::log(mat[i][j]);
        }
    }
    return result;
}

template <typename T> Vec2d<T> abs(const Vec2d<T> &mat) {
    Vec2d<T> result(mat.size(), std::vector<T>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::abs(mat[i][j]);
        }
    }
    return result;
}

template <typename T> Vec2d<T> transpose(const Vec2d<T> &mat) {
    Vec2d<T> result(mat[0].size(), std::vector<T>(mat.size(), 0.0));
    for (int i = 0; i < mat[0].size(); i++) {
        for (int j = 0; j < mat.size(); j++) {
            result[i][j] = mat[j][i];
        }
    }
    return result;
}

template <typename T>
Vec2d<T> eltwiseMult(const Vec2d<T> &a, const Vec2d<T> &b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Vec2d<T> result(a.size(), std::vector<T>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

template <typename T>
T calculateAccuracy(const Vec2d<T> &output, const Vec2d<T> &actualY) {
    T precision = 0.0028284271247461905;

    auto absMinus = abs(output - actualY);
    T count = 0.0;
    for (int i = 0; i < absMinus.size(); i++) {
        for (int j = 0; j < absMinus[i].size(); j++) {
            if (absMinus[i][j] < precision)
                count += 1.0;
        }
    }

    return count / (absMinus.size() * absMinus[0].size());

    // auto argmax = [](const Vec2d<T> &inp) {
    //     std::vector<T> out(inp.size());
    //     for (int i = 0; i < inp.size(); i++) {
    //         auto maxIter = std::max_element(inp[i].begin(), inp[i].end());
    //         out[i] = std::distance(inp[i].begin(), maxIter);
    //     }
    //     return out;
    // };

    // std::vector<T> predictions = argmax(output);

    // std::vector<T> y;
    // if (actualY.size() != 1) {
    //     y = argmax(actualY);
    // } else {
    //     for (int i = 0; i < actualY[0].size(); i++) {
    //         y.push_back(actualY[0][i]);
    //     }
    // }
    // std::vector<bool> isPredY(y.size());
    // for (int i = 0; i < y.size(); i++) {
    //     isPredY[i] = y[i] == predictions[i];
    // }
    // return std::accumulate(isPredY.begin(), isPredY.end(), 0.0) /
    //        isPredY.size();

    // predictions = (activation2.output > 0.5) * 1
    // accuracy = np.mean(predictions==y)
    // ---------------------------------------------------------------------------
    // T sum = 0.0;
    // for (int i = 0; i < output.size(); i++) {
    //     for (int j = 0; j < output[i].size(); j++) {
    //         sum += ((T)output[i][j] > 0.5) == actualY[i][j];
    //     }
    // }
    // return sum / (T)(output.size() * output[0].size());
}

// Explicit instantiations
template Vec2d<double> operator*(const double num, const Vec2d<double> &mat);
template Vec2d<float> operator*(const float num, const Vec2d<float> &mat);

template Vec2d<double> operator*(const Vec2d<double> &mat, const double num);
template Vec2d<float> operator*(const Vec2d<float> &mat, const float num);

template Vec2d<double> operator*(const Vec2d<double> &a,
                                 const Vec2d<double> &b);
template Vec2d<float> operator*(const Vec2d<float> &a, const Vec2d<float> &b);

template Vec2d<double> operator/(const Vec2d<double> &mat, const double num);
template Vec2d<float> operator/(const Vec2d<float> &mat, const float num);

template Vec2d<double> operator/(const double num, const Vec2d<double> &mat);
template Vec2d<float> operator/(const float num, const Vec2d<float> &mat);

template Vec2d<double> operator/(const Vec2d<double> &a,
                                 const Vec2d<double> &b);
template Vec2d<float> operator/(const Vec2d<float> &a, const Vec2d<float> &b);

template Vec2d<double> operator-(const Vec2d<double> &mat, const double num);
template Vec2d<float> operator-(const Vec2d<float> &mat, const float num);

template Vec2d<double> operator-(const double num, const Vec2d<double> &mat);
template Vec2d<float> operator-(const float num, const Vec2d<float> &mat);

template Vec2d<double> operator-(const Vec2d<double> &a,
                                 const Vec2d<double> &b);
template Vec2d<float> operator-(const Vec2d<float> &a, const Vec2d<float> &b);

template Vec2d<double> operator+(const Vec2d<double> &mat, const double num);
template Vec2d<float> operator+(const Vec2d<float> &mat, const float num);

template Vec2d<double> operator+(const Vec2d<double> &a,
                                 const Vec2d<double> &b);
template Vec2d<float> operator+(const Vec2d<float> &a, const Vec2d<float> &b);

template Vec2d<double> power(const Vec2d<double> &mat, const double exponent);
template Vec2d<float> power(const Vec2d<float> &mat, const float exponent);

template Vec2d<double> root(const Vec2d<double> &mat);
template Vec2d<float> root(const Vec2d<float> &mat);

template Vec2d<double> exp(const Vec2d<double> &mat);
template Vec2d<float> exp(const Vec2d<float> &mat);

template Vec2d<double> log(const Vec2d<double> &mat);
template Vec2d<float> log(const Vec2d<float> &mat);

template Vec2d<double> abs(const Vec2d<double> &mat);
template Vec2d<float> abs(const Vec2d<float> &mat);

template Vec2d<double> transpose(const Vec2d<double> &mat);
template Vec2d<float> transpose(const Vec2d<float> &mat);

template Vec2d<double> eltwiseMult(const Vec2d<double> &a,
                                   const Vec2d<double> &b);
template Vec2d<float> eltwiseMult(const Vec2d<float> &a, const Vec2d<float> &b);

template double calculateAccuracy(const Vec2d<double> &output,
                                  const Vec2d<double> &actualY);
template float calculateAccuracy(const Vec2d<float> &output,
                                 const Vec2d<float> &actualY);
