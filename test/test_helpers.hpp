#ifndef test_helpers_hpp
#define test_helpers_hpp

#include <gtest/gtest.h>
#include "utils.hpp"

template<typename T>
void populateVec2dWithSequentialData(Vec2d<T>& vec, int rows, int cols, int start = 0, int limit = 500) {
    vec.resize(rows, std::vector<T>(cols));
    int counter = start;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vec[i][j] = counter;
            counter += 1;
            if (counter > limit) counter = start;
        }
    }
}

#define VEC2D_EXPECT_EQ(A, B)                                                  \
    EXPECT_EQ(A.size(), B.size());                                             \
    EXPECT_EQ(A[0].size(), B[0].size());                                       \
    for (int i = 0; i < A.size(); i++) {                                       \
        for (int j = 0; j < A[i].size(); j++) {                                \
            EXPECT_FLOAT_EQ(A[i][j], B[i][j]);                                 \
        }                                                                      \
    }

#endif