#include "utils.hpp"

DoubleVec2d operator*(const double num, const DoubleVec2d& mat) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] * num;
        }
    }
    return result;
}

DoubleVec2d operator*(const DoubleVec2d& mat, const double num) {
    return num * mat;
}

DoubleVec2d operator*(const DoubleVec2d& a, const DoubleVec2d& b) {
    assert(a[0].size() == b.size());
    DoubleVec2d result(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            for (int p = 0; p < b.size(); p++) {
                result[i][j] += a[i][p] * b[p][j];    
            }
        }
    }
    return result;
}

DoubleVec2d operator/(const DoubleVec2d& mat, const double num) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] / num;
        }
    }
    return result;
}

DoubleVec2d operator/(const DoubleVec2d& a, const DoubleVec2d& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    DoubleVec2d result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] / b[i][j];
        }
    }
    return result;
}

DoubleVec2d operator-(const DoubleVec2d& mat, const double num) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] - num;
        }
    }
    return result;
}

DoubleVec2d operator-(const DoubleVec2d& a, const DoubleVec2d& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    DoubleVec2d result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

DoubleVec2d operator+(const DoubleVec2d& mat, const double num) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = mat[i][j] + num;
        }
    }
    return result;
}

DoubleVec2d operator+(const DoubleVec2d& a, const DoubleVec2d& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    DoubleVec2d result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

DoubleVec2d exp(const DoubleVec2d& mat, const double exponent) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::pow(mat[i][j], exponent);
        }
    }
    return result;
}

DoubleVec2d root(const DoubleVec2d& mat) {
    DoubleVec2d result(mat.size(), std::vector<double>(mat[0].size(), 0.0));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            result[i][j] = std::sqrt(mat[i][j]);
        }
    }
    return result;
}

DoubleVec2d transpose(const DoubleVec2d& mat) {
    DoubleVec2d result(mat[0].size(), std::vector<double>(mat.size(), 0.0));
    for (int i = 0; i < mat[0].size(); i++) {
        for (int j = 0; j < mat.size(); j++) {
            result[i][j] = mat[j][i];
        }
    }
    return result;
}
