#include <gtest/gtest.h>

#define VEC2D_EXPECT_EQ(A, B)                                                  \
    EXPECT_EQ(A.size(), B.size());                                             \
    EXPECT_EQ(A[0].size(), B[0].size());                                       \
    for (int i = 0; i < A.size(); i++) {                                       \
        for (int j = 0; j < A[i].size(); j++) {                                \
            EXPECT_EQ(A[i][j], B[i][j]);                                       \
        }                                                                      \
    }