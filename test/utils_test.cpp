#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "utils.hpp"

typedef ::testing::Types<float, double> TestTypes;

template <typename T>
class UtilsTest : public ::testing::Test {};

TYPED_TEST_SUITE(UtilsTest, TestTypes);

TYPED_TEST(UtilsTest, UtilsTestMultSmall) {
    Vec2d<TypeParam> a {{4, 9}, {3, 4}};
    Vec2d<TypeParam> b {{4, 9, 5}, {3, 4, 6}};

    auto result = a * b;

    Vec2d<TypeParam> expected {{43, 72, 74}, {24, 43, 39}};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsTest, UtilsTestMultMed) {
    Vec2d<TypeParam> a {{9, 3, 0, 2, 4,},
                        {3, 0, 0, 3, 2,},
                        {1, 2, 5, 8, 5,},
                        {8, 3, 5, 3, 1,},
                        {1, 0, 3, 2, 8,},
                        {6, 1, 3, 3, 1,},
                        {0, 0, 9, 7, 2,},
                        {9, 6, 1, 4, 0,},
                        {4, 7, 5, 8, 6,},
                        {6, 1, 9, 4, 0,},
                        {8, 1, 9, 7, 5,},
                        {6, 5, 5, 2, 4,},
                        {9, 2, 4, 1, 0,},
                        {3, 5, 9, 5, 0,},
                        {1, 5, 3, 7, 4,},};

    Vec2d<TypeParam> b {{5, 4, 0, 4, 6, 5, 1, 1, 1, 5, 4, 6, 0, 5, 0, 1, 8, 6, 9, 8,},
                        {9, 4, 5, 9, 4, 0, 0, 7, 2, 8, 3, 4, 2, 0, 8, 7, 6, 8, 4, 5,},
                        {1, 2, 8, 0, 7, 4, 4, 4, 1, 9, 6, 5, 2, 6, 2, 4, 8, 6, 3, 3,},
                        {4, 8, 2, 5, 9, 8, 1, 9, 7, 1, 6, 1, 1, 6, 2, 7, 2, 8, 2, 1,},
                        {2, 3, 3, 7, 1, 4, 6, 4, 8, 2, 9, 4, 9, 1, 2, 8, 0, 1, 4, 8,},
                        {8, 0, 8, 0, 0, 0, 6, 6, 5, 7, 0, 0, 4, 7, 2, 0, 4, 0, 7, 0,},
                        {1, 1, 8, 0, 3, 7, 0, 9, 8, 1, 9, 6, 6, 2, 5, 9, 5, 5, 6, 9,},
                        {8, 8, 2, 0, 6, 9, 6, 6, 4, 1, 1, 5, 3, 2, 0, 9, 2, 2, 2, 0,},
                        {4, 9, 0, 6, 2, 1, 7, 0, 8, 0, 9, 3, 4, 4, 7, 0, 5, 5, 7, 2,},
                        {8, 0, 3, 4, 5, 5, 0, 5, 1, 8, 8, 1, 1, 0, 1, 7, 8, 2, 3, 8,},
                        {4, 0, 5, 8, 2, 8, 5, 8, 9, 6, 9, 3, 6, 8, 7, 7, 6, 8, 1, 5,},
                        {9, 2, 0, 4, 4, 9, 1, 3, 1, 3, 6, 1, 6, 6, 1, 3, 2, 7, 8, 9,},
                        {6, 9, 4, 0, 2, 9, 3, 6, 9, 3, 5, 7, 2, 0, 1, 9, 7, 6, 6, 1,},
                        {6, 3, 6, 1, 0, 4, 9, 2, 6, 9, 8, 7, 6, 3, 1, 3, 1, 7, 1, 5,},
                        {9, 8, 5, 6, 2, 9, 5, 2, 9, 3, 5, 4, 2, 2, 3, 0, 1, 8, 5, 6,},
                        {3, 8, 1, 3, 5, 8, 3, 2, 5, 0, 9, 2, 1, 5, 5, 9, 9, 9, 2, 8,},
                        {1, 5, 8, 5, 1, 6, 2, 6, 8, 3, 9, 7, 0, 4, 0, 4, 4, 8, 8, 3,},
                        {5, 0, 3, 4, 1, 6, 8, 1, 7, 8, 1, 1, 3, 3, 4, 1, 2, 6, 8, 3,},
                        {8, 5, 8, 0, 7, 7, 6, 9, 5, 5, 8, 4, 2, 1, 1, 2, 5, 8, 4, 3,},
                        {7, 1, 3, 0, 0, 0, 0, 3, 5, 7, 7, 7, 3, 8, 9, 0, 5, 4, 4, 4,},};

    auto result = a * b;

    Vec2d<TypeParam> expected {{88,  76,  31, 101,  88,  77, 35,  64,  61,  79,  93,  84, 44,  61, 36,  76,  94,  98, 113, 121,},
                               {31,  42,  12,  41,  47,  47, 18,  38,  40,  22,  48,  29, 21,  35, 10,  40,  30,  44,  41,  43,},
                               {70, 101,  81,  97, 126, 109, 59, 127, 106,  84, 133,  67, 67,  88, 52, 131,  76, 121,  68,  81,},
                               {86,  81,  64,  81, 123,  88, 37,  80,  48, 114,  98,  92, 28,  89, 42,  78, 128, 127, 109, 105,},
                               {32,  50,  52,  70,  53,  65, 63,  63,  82,  50, 106,  55, 80,  43, 26,  91,  36,  48,  54,  83,},
                               {56,  61,  38,  55,  89,  70, 27,  56,  40,  70,  72,  62, 20,  67, 22,  54,  84,  87,  77,  73,},
                               {41,  80,  92,  49, 128, 100, 55, 107,  74,  92, 114,  60, 43,  98, 36, 101,  86, 112,  49,  50,},
                               {116,  94,  46, 110, 121,  81, 17,  91,  50, 106,  84,  87, 18,  75, 58,  83, 124, 140, 116, 109,},
                               {132, 136, 109, 161, 165, 128, 68, 169, 127, 141, 169, 109, 86, 104, 94, 177, 130, 180, 119, 138,},
                               {64,  78,  85,  53, 139,  98, 46,  85,  45, 123, 105,  89, 24, 108, 34,  77, 134, 130,  93,  84,},
                               {96, 125, 106, 111, 183, 152, 81, 134, 108, 146, 176, 124, 72, 141, 50, 140, 156, 171, 137, 143,},
                               {96,  82,  81, 107, 113,  82, 52,  95,  67, 125, 117,  99, 58,  76, 62, 107, 122, 126, 109, 122,},
                               {71,  60,  44,  59,  99,  69, 26,  48,  24,  98,  72,  83, 13,  75, 26,  46, 118, 102, 103,  95,},
                               {89,  90, 107,  82, 146,  91, 44, 119,  57, 141, 111,  88, 33,  99, 68, 109, 136, 152,  84,  81,},
                               {89,  98,  75, 112, 114,  89, 44, 127,  95,  87, 115,  64, 59,  69, 68, 129,  76, 124,  68,  81,},};

    VEC2D_EXPECT_EQ(result, expected)
}

TYPED_TEST(UtilsTest, UtilsTestMultBig) {
    Vec2d<TypeParam> a {{9, 4, 7, 8, 3, 8, 0, 0, 6, 8, 9, 1, 7, 9, 7, 7, 0, 7, 1, 7},
                        {0, 2, 2, 6, 4, 6, 6, 3, 6, 6, 7, 1, 1, 2, 1, 4, 5, 2, 6, 4},
                        {1, 3, 4, 5, 9, 3, 0, 9, 2, 1, 1, 6, 4, 3, 6, 2, 0, 3, 8, 2},
                        {9, 1, 6, 7, 4, 0, 1, 5, 4, 9, 4, 7, 8, 8, 8, 3, 5, 5, 8, 2},
                        {9, 4, 2, 7, 4, 0, 5, 9, 8, 7, 1, 8, 9, 9, 2, 4, 7, 9, 4, 3},
                        {7, 1, 2, 8, 0, 1, 7, 6, 0, 7, 9, 4, 6, 0, 3, 1, 8, 5, 2, 5},
                        {4, 0, 7, 9, 5, 9, 8, 1, 4, 1, 1, 9, 4, 2, 8, 1, 2, 4, 4, 9},
                        {0, 0, 0, 9, 4, 1, 8, 8, 2, 3, 3, 6, 3, 5, 4, 0, 3, 3, 0, 1},
                        {5, 3, 1, 1, 9, 1, 6, 1, 5, 6, 6, 1, 0, 1, 6, 5, 4, 1, 9, 9},
                        {6, 1, 9, 3, 0, 7, 3, 8, 2, 0, 4, 5, 4, 9, 4, 7, 8, 1, 7, 3},
                        {0, 1, 9, 7, 0, 5, 2, 5, 6, 2, 7, 3, 4, 9, 2, 1, 5, 7, 6, 2},
                        {4, 3, 0, 9, 6, 9, 5, 2, 1, 8, 3, 5, 7, 5, 4, 9, 8, 3, 9, 7},
                        {7, 2, 5, 8, 5, 6, 5, 7, 3, 6, 8, 7, 7, 4, 6, 9, 1, 6, 3, 1},
                        {0, 4, 9, 5, 0, 7, 8, 2, 9, 0, 4, 3, 7, 8, 0, 8, 7, 3, 9, 1},
                        {7, 8, 0, 7, 8, 3, 4, 3, 5, 4, 8, 0, 3, 0, 7, 3, 3, 4, 6, 6},
                        {2, 2, 8, 9, 2, 0, 7, 7, 3, 7, 6, 8, 1, 4, 0, 4, 0, 1, 1, 8},
                        {5, 9, 2, 8, 2, 4, 2, 9, 8, 5, 9, 9, 9, 0, 8, 0, 8, 3, 7, 0},
                        {1, 2, 1, 8, 5, 4, 5, 2, 2, 8, 7, 5, 7, 8, 3, 7, 5, 5, 6, 5},
                        {2, 8, 1, 9, 8, 6, 7, 9, 9, 0, 8, 0, 0, 5, 2, 8, 3, 9, 5, 7},
                        {4, 4, 0, 9, 8, 7, 2, 6, 0, 3, 1, 8, 3, 6, 9, 2, 5, 9, 5, 1},
                        {3, 9, 1, 2, 4, 5, 5, 5, 1, 0, 0, 8, 2, 7, 2, 8, 5, 8, 1, 9},
                        {1, 8, 7, 0, 5, 9, 0, 5, 1, 8, 6, 6, 9, 9, 7, 5, 7, 9, 3, 1},
                        {3, 1, 7, 7, 5, 5, 3, 8, 5, 8, 1, 4, 3, 8, 3, 3, 2, 1, 7, 2},
                        {4, 6, 0, 5, 8, 7, 8, 3, 9, 6, 4, 6, 0, 0, 6, 4, 2, 8, 7, 1},
                        {3, 5, 1, 6, 9, 1, 4, 3, 4, 1, 4, 3, 5, 5, 4, 6, 4, 3, 0, 1},
                        {1, 4, 4, 7, 4, 3, 1, 5, 1, 7, 6, 1, 6, 8, 7, 3, 8, 9, 1, 1},
                        {9, 3, 1, 2, 7, 6, 9, 1, 4, 4, 7, 6, 5, 5, 5, 6, 8, 7, 0, 7},
                        {9, 1, 3, 0, 9, 2, 4, 9, 5, 2, 8, 2, 6, 2, 2, 1, 9, 7, 9, 2},
                        {0, 0, 1, 7, 3, 9, 3, 8, 8, 8, 6, 8, 9, 7, 8, 8, 9, 1, 7, 6},
                        {4, 4, 8, 4, 4, 5, 7, 1, 9, 9, 8, 2, 7, 8, 3, 5, 8, 4, 4, 5},
                        {2, 6, 1, 6, 6, 6, 3, 9, 5, 0, 0, 7, 1, 1, 2, 9, 1, 6, 6, 1},
                        {1, 9, 5, 2, 3, 5, 0, 6, 5, 9, 6, 4, 1, 0, 1, 1, 7, 7, 1, 2},
                        {6, 4, 4, 4, 3, 2, 2, 1, 7, 6, 4, 0, 6, 8, 0, 1, 4, 4, 5, 8},
                        {3, 1, 9, 0, 5, 8, 7, 4, 7, 6, 1, 9, 7, 1, 3, 1, 3, 8, 5, 1},
                        {3, 6, 3, 3, 4, 8, 7, 9, 4, 6, 9, 4, 0, 0, 5, 8, 0, 8, 5, 5},
                        {4, 8, 1, 2, 4, 2, 0, 3, 2, 0, 3, 9, 6, 5, 2, 5, 6, 7, 6, 5},
                        {9, 5, 9, 0, 8, 5, 3, 0, 5, 9, 2, 6, 0, 2, 3, 5, 1, 9, 0, 6},
                        {6, 1, 2, 4, 2, 5, 1, 8, 1, 7, 4, 8, 9, 5, 6, 4, 9, 4, 5, 8},
                        {4, 4, 5, 7, 4, 7, 9, 5, 5, 1, 2, 8, 7, 8, 8, 2, 4, 9, 3, 0},
                        {3, 8, 7, 4, 5, 0, 4, 9, 7, 0, 8, 6, 6, 2, 6, 5, 1, 7, 2, 7},
                        {1, 6, 0, 3, 7, 8, 5, 7, 1, 3, 4, 7, 7, 4, 8, 1, 1, 5, 8, 9},
                        {9, 9, 9, 7, 5, 7, 1, 4, 7, 5, 3, 5, 0, 5, 7, 3, 6, 0, 7, 7},
                        {6, 3, 3, 8, 2, 3, 8, 6, 4, 6, 8, 9, 8, 8, 1, 7, 7, 1, 3, 1},
                        {9, 5, 4, 3, 2, 9, 5, 2, 5, 8, 5, 5, 6, 9, 5, 5, 0, 3, 5, 1},
                        {2, 4, 1, 3, 2, 0, 3, 9, 0, 2, 3, 8, 0, 9, 0, 0, 5, 6, 2, 3},
                        {9, 3, 4, 0, 4, 5, 7, 0, 5, 0, 0, 6, 5, 4, 0, 5, 7, 1, 3, 8},
                        {7, 2, 8, 3, 2, 3, 8, 7, 6, 1, 4, 9, 9, 1, 0, 0, 1, 8, 3, 7},
                        {8, 5, 5, 5, 4, 4, 5, 3, 1, 2, 6, 5, 5, 4, 4, 5, 1, 9, 7, 2},
                        {5, 1, 7, 9, 0, 7, 6, 7, 9, 9, 3, 2, 6, 4, 3, 1, 7, 5, 6, 7},
                        {0, 9, 6, 9, 8, 5, 5, 7, 4, 3, 2, 4, 6, 4, 3, 1, 3, 0, 7, 5},};

    Vec2d<TypeParam> b {{26,2,0,36,17,5,24,37,3,12,25,28,47,26,49,18,43,3,41,39,34,31,32,19,3,25,10,19,10,40,42,12,48,2,11,2,49,10,11,22,28,39,49,32,27,6,42,26,32,32},
                        {36,21,21,19,45,15,44,8,39,31,49,49,46,43,16,39,41,41,7,12,27,20,40,19,25,44,22,13,36,20,42,32,1,13,19,33,4,36,24,30,19,47,14,46,27,41,28,28,1,12},
                        {29,21,11,13,25,4,4,7,50,23,44,11,4,47,38,4,39,13,42,28,11,45,37,32,29,28,20,20,39,12,28,33,25,12,10,37,39,49,21,29,40,4,9,16,21,40,22,39,25,10},
                        {16,20,7,27,44,1,12,13,44,25,6,4,24,45,12,9,30,49,19,12,49,25,27,7,11,9,42,48,4,7,6,5,27,12,14,11,49,11,17,33,15,12,16,2,44,6,11,33,15,30},
                        {42,22,25,35,1,13,3,20,48,35,30,44,30,34,44,37,42,11,12,46,38,40,43,49,11,47,9,7,7,11,39,50,10,20,35,49,31,8,8,17,8,29,15,21,29,44,7,46,0,46},
                        {1,21,37,17,14,46,31,30,40,19,26,2,45,12,34,27,33,12,8,14,4,0,41,37,29,27,24,49,38,6,43,6,7,12,26,26,15,20,10,25,20,20,12,3,45,36,18,28,4,43},
                        {35,5,30,4,16,40,49,22,46,26,0,4,33,12,8,21,37,40,28,37,34,1,35,9,45,16,19,50,36,35,14,31,27,29,7,17,44,19,43,17,44,42,34,31,19,47,40,1,26,43},
                        {14,47,9,35,32,16,0,15,14,25,49,50,16,3,35,46,14,24,46,35,42,11,45,14,12,38,0,11,47,47,43,34,15,34,44,42,13,19,5,50,36,26,26,3,16,23,6,44,29,43},
                        {12,50,25,13,4,15,25,33,2,33,9,12,43,27,41,33,39,45,16,8,33,19,3,28,34,43,22,4,17,18,30,49,33,49,4,23,30,20,23,40,3,26,14,31,36,4,13,2,10,0},
                        {5,14,28,37,22,11,44,15,11,3,41,42,49,40,45,16,17,6,12,40,19,21,18,27,2,35,22,23,25,48,6,14,25,12,48,29,12,5,14,10,16,37,32,3,19,24,36,50,36,32},
                        {37,27,37,42,38,8,49,11,8,21,36,17,45,38,2,39,42,38,12,12,4,31,40,19,4,38,31,37,14,41,10,15,24,14,0,18,26,37,9,35,33,10,14,32,44,5,6,37,14,22},
                        {7,15,33,31,50,33,6,14,45,40,47,4,32,15,36,1,14,36,50,27,46,38,13,32,5,8,26,22,2,49,40,19,27,44,48,5,38,4,35,38,39,18,43,43,10,19,38,31,23,37},
                        {35,27,5,18,8,34,13,3,39,31,16,37,5,10,28,1,19,30,32,48,42,8,7,27,48,28,49,14,10,36,38,31,5,21,14,36,44,3,5,2,8,15,20,39,8,41,2,30,5,35},
                        {31,32,10,35,43,21,32,31,6,37,36,15,44,5,23,33,20,33,30,34,40,22,34,22,3,13,39,13,17,30,21,32,25,40,45,25,31,20,19,19,46,21,11,2,29,41,42,9,7,33},
                        {34,22,31,36,32,1,8,17,49,39,39,27,40,39,48,12,11,7,38,26,21,23,44,2,36,24,37,17,49,24,6,34,14,12,24,32,9,50,30,16,17,12,29,49,16,8,27,46,33,26},
                        {40,38,29,3,38,36,45,37,15,24,17,34,23,5,35,44,12,29,10,21,13,29,22,23,29,5,46,25,13,42,17,16,1,14,24,6,30,7,9,9,48,48,36,28,37,32,4,17,4,16},
                        {44,3,2,32,40,35,30,41,28,46,14,41,3,44,41,13,44,36,29,30,41,40,5,44,44,30,12,22,0,31,13,8,40,8,24,48,4,19,26,46,49,3,29,6,3,34,1,21,22,38},
                        {40,22,41,30,16,25,11,6,12,48,10,26,41,43,21,5,34,33,32,18,25,22,29,33,24,34,18,31,36,43,41,49,6,10,22,37,48,7,24,16,19,45,11,1,11,38,35,4,26,16},
                        {16,49,24,32,28,43,18,31,48,23,47,8,25,36,25,39,27,13,32,0,47,20,39,44,27,19,28,44,31,8,49,9,8,34,42,25,44,9,12,25,10,36,0,39,18,36,31,20,17,35},
                        {11,24,25,44,34,13,43,11,23,8,11,15,16,5,32,44,43,0,42,13,31,23,2,14,16,32,16,39,16,1,23,49,16,25,19,38,35,29,12,22,26,8,14,22,41,37,23,34,31,33}};

    auto result = a * b;

    Vec2d<TypeParam> expected {{2710, 2581, 2359, 2995, 2878, 1826, 2868, 2077, 2634, 2707, 2850, 2386, 3660, 2927, 3311, 2501, 3256, 2502, 2673, 2632, 2792, 2546, 2988, 2492, 2069, 2911, 3027, 2766, 2346, 2871, 2722, 2815, 2081, 1934, 2294, 2682, 3368, 2207, 1691, 2360, 2644, 2591, 2207, 2258, 3141, 2743, 2401, 3039, 1889, 2833,}, 
                               {1751, 1891, 1753, 1946, 1970, 1661, 2140, 1601, 2075, 1882, 1855, 1548, 2315, 2043, 2066, 2022, 2354, 1981, 1655, 1610, 2118, 1621, 2030, 1924, 1647, 2058, 1813, 2201, 1606, 1850, 1816, 1767, 1443, 1591, 1688, 1987, 2124, 1372, 1284, 1961, 1872, 1820, 1411, 1462, 2095, 2032, 1416, 1955, 1262, 2191,}, 
                               {1732, 2082, 1531, 2085, 1948, 1439, 1080, 1304, 2475, 2110, 2412, 1793, 2035, 1885, 2333, 1829, 1926, 1646, 2129, 1820, 2454, 1739, 2333, 1922, 1442, 1990, 1709, 1657, 1778, 1799, 2339, 2053, 1104, 1715, 2134, 2134, 2178, 1315, 1145, 1895, 1622, 1748, 1364, 1695, 1695, 2056, 1439, 2299, 1214, 2285,}, 
                               {2609, 2499, 1954, 3074, 2902, 1911, 2157, 2084, 2863, 2864, 3120, 2510, 3144, 2942, 3428, 2116, 2850, 2427, 3112, 2833, 3410, 2642, 2828, 2603, 2028, 2660, 2717, 2420, 2157, 3108, 2788, 2568, 2237, 2172, 2767, 2696, 3297, 1809, 1796, 2501, 2617, 2469, 2362, 2349, 2311, 2638, 2433, 3039, 2106, 3063,}, 
                               {2828, 2679, 2050, 3058, 3000, 2307, 2457, 2273, 2728, 3230, 2906, 2839, 3351, 2779, 3496, 2409, 3175, 3046, 3259, 3080, 3873, 2581, 2760, 2775, 2296, 2946, 2668, 2485, 2201, 3517, 3205, 3034, 2380, 2537, 2882, 2925, 3547, 1625, 2013, 2828, 2914, 2946, 2610, 2281, 2429, 3032, 2540, 2767, 2107, 3271,}, 
                               {2121, 1594, 1607, 2438, 2388, 1568, 2156, 1500, 2178, 2103, 2020, 1978, 2355, 2321, 2298, 1675, 2539, 2153, 2384, 2198, 2556, 1909, 2117, 1810, 1709, 2186, 1935, 2391, 1639, 2628, 1916, 1834, 1907, 1435, 1756, 2118, 2544, 1468, 1480, 2164, 2308, 1850, 2031, 1687, 1906, 2024, 1703, 2391, 1853, 2591,}, 
                               {2042, 1989, 2115, 2424, 2430, 1929, 1953, 1735, 3250, 2511, 2251, 1469, 2686, 2352, 2861, 1827, 2808, 2126, 2763, 2212, 2841, 2097, 2496, 2186, 2140, 2233, 2314, 2747, 2077, 2058, 2503, 2471, 1841, 1983, 2082, 2349, 3096, 1853, 1864, 2268, 2303, 1940, 1971, 2137, 2412, 2556, 2165, 2549, 1805, 2866,}, 
                               {1556, 1446, 1298, 1764, 1855, 1244, 1305, 1135, 1925, 1894, 1590, 1381, 1876, 1528, 1710, 1343, 1696, 1937, 1846, 1791, 2255, 1338, 1825, 1310, 1267, 1550, 1545, 1690, 1364, 1943, 1485, 1679, 1351, 1534, 1660, 1664, 1952, 1061, 1296, 1742, 1766, 1439, 1485, 1168, 1474, 1648, 1348, 1748, 1244, 2138,}, 
                               {2099, 1966, 1911, 2346, 2029, 1592, 2294, 1787, 2295, 1967, 2164, 1936, 2562, 2214, 2563, 2377, 2603, 1614, 2013, 1885, 2374, 1981, 2229, 2065, 1702, 2334, 1804, 2133, 1721, 1946, 2060, 2227, 1551, 1673, 1895, 2213, 2342, 1579, 1386, 1883, 1919, 2127, 1696, 2115, 2149, 2229, 1758, 2285, 1536, 2404,}, 
                               {2304, 2314, 1662, 2387, 2763, 2107, 2093, 2110, 2582, 2528, 2731, 1950, 2429, 2120, 2876, 2228, 2598, 2201, 2760, 2268, 2733, 2205, 2641, 2342, 2059, 2155, 2286, 2346, 2090, 2511, 2592, 2059, 1880, 1963, 2310, 2372, 2685, 1877, 1552, 2489, 2894, 1987, 1962, 1885, 2208, 2617, 1896, 2410, 1649, 2748,}, 
                               {2061, 2251, 1715, 2244, 2393, 1712, 1838, 1595, 2265, 2435, 2318, 1567, 2391, 2348, 2277, 1842, 2511, 2366, 2327, 1833, 2476, 1969, 2345, 2156, 1811, 2182, 2178, 2225, 1942, 2160, 2238, 2141, 1672, 1887, 1932, 2318, 2619, 1743, 1481, 2306, 2268, 1718, 1363, 1456, 2103, 2303, 1735, 2067, 1456, 2240,}, 
                               {2607, 2487, 2302, 2901, 3018, 2594, 2856, 2354, 3257, 2796, 2736, 2447, 3160, 2688, 3277, 2609, 3081, 2553, 2656, 2640, 3341, 2392, 2800, 2816, 2384, 2610, 2884, 3105, 2078, 2774, 2810, 2359, 1913, 2099, 2826, 2782, 3196, 1640, 1772, 2451, 2756, 2692, 2327, 2251, 2765, 3157, 2197, 2964, 1807, 3461,}, 
                               {2703, 2563, 2350, 2788, 2893, 2131, 2456, 2045, 2953, 2846, 2949, 2458, 3335, 2700, 3196, 2395, 2921, 2700, 2820, 2794, 3044, 2459, 3129, 2503, 2133, 2677, 2817, 2755, 2342, 3252, 2844, 2587, 1985, 2090, 2526, 2536, 3345, 1858, 1776, 2525, 2821, 2728, 2481, 2357, 2731, 2727, 2209, 3041, 1904, 3118,}, 
                               {2473, 2558, 1932, 2003, 2571, 2455, 2475, 2119, 2815, 2734, 2366, 1752, 2570, 2362, 2590, 2266, 2866, 2817, 2388, 2078, 2821, 2062, 2483, 2548, 2571, 2284, 2627, 2571, 2091, 2359, 2627, 2300, 1762, 2229, 2023, 2426, 2982, 1811, 1775, 2426, 2675, 2285, 1702, 2080, 2387, 2925, 1874, 1970, 1352, 2517,}, 
                               {2396, 2128, 1991, 2581, 2327, 1553, 2340, 1748, 2552, 2322, 2373, 2271, 2974, 2673, 2621, 2417, 2921, 2145, 2121, 2041, 2651, 2074, 2654, 2091, 1868, 2683, 2122, 2341, 1976, 2207, 2394, 2387, 1648, 1628, 1865, 2414, 2595, 1801, 1482, 2203, 1930, 2355, 1833, 2271, 2466, 2252, 1814, 2597, 1562, 2553,}, 
                               {1739, 1902, 1699, 2182, 2500, 1367, 2044, 1344, 2254, 1898, 2236, 1595, 2302, 1970, 2231, 1928, 2343, 2120, 2341, 2012, 2457, 1983, 2123, 1693, 1343, 1990, 1930, 2221, 1686, 2286, 1899, 2112, 1740, 1863, 1931, 1922, 2614, 1573, 1493, 2171, 2377, 1823, 1817, 1607, 2189, 2111, 1818, 2380, 1676, 2330,}, 
                               {2612, 2662, 2176, 3020, 3096, 2124, 2343, 2004, 3179, 3100, 3258, 2695, 3268, 3166, 3213, 2360, 3121, 3028, 2920, 2541, 3452, 2466, 2941, 2612, 2418, 3077, 2658, 2562, 2357, 3142, 3037, 2539, 2151, 2290, 2516, 2869, 2906, 2082, 1935, 3084, 2498, 2465, 2359, 2770, 2444, 2499, 2058, 3162, 1943, 3076,}, 
                               {2476, 2330, 2143, 2664, 2769, 2151, 2585, 1912, 2690, 2611, 2493, 2172, 2933, 2456, 2706, 2293, 2714, 2546, 2390, 2421, 2962, 2205, 2525, 2427, 1980, 2385, 2699, 2665, 1846, 2729, 2353, 2327, 1745, 2029, 2480, 2495, 2969, 1540, 1639, 2187, 2539, 2369, 1993, 1970, 2466, 2775, 2013, 2592, 1619, 2948,}, 
                               {2822, 2927, 2490, 2763, 2872, 2177, 2798, 2170, 2759, 2982, 2584, 2484, 3413, 2727, 2836, 3189, 3421, 3034, 2480, 2237, 3155, 2304, 3170, 2537, 2271, 3062, 2476, 2862, 2485, 2658, 2974, 3087, 1794, 2317, 2305, 2905, 3173, 2044, 1802, 2865, 2688, 2923, 1950, 2139, 3100, 2974, 2022, 2575, 1636, 2939,}, 
                               {2345, 2105, 2041, 2709, 2642, 1933, 1723, 1816, 2940, 2861, 2656, 2199, 2963, 2570, 2884, 1953, 2540, 2308, 2568, 2359, 3043, 2157, 2848, 2352, 1845, 2360, 2273, 2378, 2095, 2533, 2667, 2336, 1630, 1840, 2637, 2538, 2704, 1548, 1709, 2322, 2265, 2312, 1995, 1866, 2135, 2541, 2102, 2645, 1649, 2984,}, 
                               {2279, 1961, 1944, 2232, 2634, 2037, 2250, 1712, 2381, 2531, 2246, 2105, 2590, 1838, 2522, 2235, 2538, 2240, 2381, 2080, 2648, 1962, 2261, 2104, 1837, 2181, 2002, 2154, 1871, 2402, 2570, 2487, 1387, 1852, 2247, 2308, 2464, 1522, 1646, 2125, 2569, 2362, 1914, 1863, 2102, 2785, 1994, 2048, 1420, 2530,}, 
                               {2881, 2530, 2394, 2907, 2966, 2355, 2497, 1953, 3037, 3151, 3355, 2846, 3292, 2898, 3326, 2324, 2958, 2591, 2740, 2834, 2882, 2499, 3087, 2914, 2363, 3007, 2754, 2427, 2580, 3171, 3037, 2796, 1705, 2037, 2868, 3192, 2734, 2106, 1821, 2492, 2838, 2578, 2127, 2206, 2377, 3292, 2255, 3070, 1745, 3062,}, 
                               {1863, 2305, 1675, 2343, 2330, 1714, 1832, 1793, 2489, 2216, 2636, 1917, 2575, 2203, 2777, 2098, 2363, 2024, 2375, 2241, 2748, 1984, 2527, 2210, 1651, 2231, 2107, 2126, 2020, 2259, 2377, 2152, 1726, 2012, 2442, 2328, 2588, 1534, 1426, 2218, 2206, 2075, 1733, 1582, 2174, 2402, 1924, 2511, 1584, 2630,}, 
                               {2288, 2303, 2453, 2422, 2290, 2074, 2319, 1975, 2791, 2669, 2496, 2085, 3361, 2716, 2859, 2328, 2884, 2461, 2227, 2134, 2796, 2052, 2791, 2450, 2085, 2674, 2170, 2521, 2265, 2549, 2714, 2544, 1716, 2044, 2267, 2401, 2792, 1620, 1859, 2367, 2094, 2778, 2024, 2264, 2416, 2500, 2196, 2385, 1648, 2696,}, 
                               {2179, 1712, 1471, 1870, 1941, 1396, 1703, 1465, 2084, 2192, 1820, 1927, 2205, 1901, 2145, 1789, 2169, 2070, 1730, 1986, 2297, 1785, 2049, 1773, 1579, 1973, 1887, 1601, 1347, 2023, 1884, 2043, 1326, 1485, 1647, 1953, 2145, 1278, 1274, 1740, 1867, 1861, 1606, 1681, 1857, 2026, 1323, 1943, 1025, 2108,}, 
                               {2505, 2007, 1787, 2529, 2519, 1619, 2012, 1593, 2340, 2629, 2409, 2362, 2636, 2618, 2581, 1831, 2486, 2331, 2244, 2328, 2562, 2105, 2496, 2149, 1871, 2427, 2292, 2096, 1978, 2592, 2084, 2273, 1612, 1547, 2193, 2637, 2358, 1716, 1517, 2119, 2315, 1997, 1742, 1515, 2005, 2452, 1736, 2502, 1580, 2514,}, 
                               {2850, 2028, 2315, 2783, 2651, 2239, 2832, 2200, 2693, 2862, 2393, 2397, 3287, 2533, 3148, 2373, 3263, 2523, 2706, 2774, 2959, 2413, 2662, 2553, 2264, 2759, 2422, 2665, 1993, 3024, 2677, 2748, 2146, 1964, 2236, 2644, 3097, 1829, 1939, 2401, 2878, 2600, 2496, 2338, 2556, 2899, 2288, 2580, 1877, 3121,}, 
                               {2605, 2287, 1809, 2726, 2158, 1999, 1972, 1999, 2397, 2644, 2582, 2481, 2646, 2556, 2915, 2340, 3007, 2180, 2647, 2473, 2973, 2265, 2673, 2657, 2034, 2831, 1841, 2168, 1968, 2722, 2896, 2490, 1931, 1904, 2183, 2756, 2834, 1570, 1451, 2498, 2330, 2319, 1945, 2051, 2017, 2570, 1779, 2489, 1726, 2882,}, 
                               {2645, 3084, 2512, 3177, 3311, 2700, 2847, 2510, 3267, 3199, 3169, 2634, 3352, 2714, 3686, 2833, 3079, 2970, 3082, 2851, 3589, 2543, 2889, 2928, 2661, 2947, 3156, 2994, 2404, 3241, 2940, 2764, 2190, 2694, 3091, 3121, 3169, 2031, 1957, 2989, 3031, 2531, 2513, 2486, 2899, 3063, 2141, 3275, 2031, 3585,}, 
                               {2865, 2566, 2297, 2821, 2832, 2264, 3054, 2271, 2865, 2912, 2800, 2475, 3334, 2997, 3282, 2554, 3439, 2831, 2708, 2774, 3085, 2575, 2779, 2821, 2519, 3040, 2797, 2746, 2289, 2981, 2696, 2862, 2289, 2268, 2376, 2989, 3210, 2160, 1973, 2641, 2910, 2580, 2225, 2309, 2783, 3122, 2307, 2800, 1908, 2997,}, 
                               {1876, 2220, 1833, 1882, 2184, 1871, 1619, 1646, 2360, 2297, 2225, 1883, 2401, 1867, 2379, 2136, 2153, 2160, 2011, 1739, 2499, 1732, 2365, 2039, 1689, 2015, 1851, 1980, 1859, 2147, 2551, 2072, 1168, 1812, 2119, 1971, 2350, 1205, 1350, 2118, 1981, 2335, 1678, 1747, 2016, 2216, 1549, 2010, 1184, 2276,}, 
                               {1832, 1746, 1743, 2111, 2112, 1441, 1924, 1331, 1934, 2075, 2339, 2135, 2427, 2410, 2319, 1786, 2399, 1985, 1787, 1766, 2007, 1854, 2058, 2068, 1525, 2461, 1539, 1750, 1819, 2233, 2120, 1983, 1415, 1397, 1867, 2309, 1751, 1531, 1335, 2161, 1893, 1907, 1520, 1420, 1785, 2074, 1512, 2243, 1397, 1985,}, 
                               {1956, 1960, 1496, 2266, 2031, 1507, 2165, 1612, 1873, 2002, 2010, 1826, 2449, 2103, 2390, 1985, 2598, 1908, 2086, 1938, 2477, 1847, 1906, 2034, 1613, 2286, 1959, 1939, 1553, 1978, 2174, 2220, 1673, 1709, 1783, 2209, 2522, 1449, 1277, 1887, 1895, 1909, 1475, 1622, 2094, 2267, 1805, 1999, 1402, 2178,}, 
                               {2069, 2092, 2110, 2169, 1981, 2150, 1803, 1671, 2813, 2553, 2492, 1860, 2656, 2366, 2922, 1674, 2633, 2137, 2581, 2389, 2618, 1996, 2404, 2559, 2195, 2482, 2028, 2232, 2203, 2554, 2802, 2506, 1700, 2023, 2214, 2492, 2834, 1544, 1751, 2181, 2192, 2246, 1909, 1991, 1897, 2703, 2092, 2327, 1694, 2614,}, 
                               {2360, 2511, 2560, 2561, 2636, 2067, 2609, 1844, 2627, 2490, 2783, 2292, 3272, 2489, 2781, 2702, 2864, 2375, 2410, 2174, 2509, 2060, 3043, 2290, 2001, 2751, 2229, 2731, 2606, 2827, 2725, 2584, 1578, 1958, 2301, 2516, 2746, 1933, 1680, 2471, 2549, 2763, 2071, 2162, 2644, 2648, 2090, 2706, 1816, 2768,}, 
                               {2191, 1966, 1707, 2257, 2448, 1888, 1860, 1576, 2301, 2470, 2346, 2036, 2355, 2051, 2404, 1920, 2380, 2148, 2288, 1902, 2656, 2023, 2058, 2207, 1720, 2124, 2014, 1888, 1540, 2307, 2569, 2159, 1357, 1757, 2120, 2172, 2424, 1330, 1432, 2050, 2139, 2086, 1702, 2029, 1799, 2413, 1752, 2045, 1277, 2314,}, 
                               {2176, 1793, 2107, 2340, 2083, 1561, 2153, 1677, 2297, 2234, 2448, 2112, 2982, 2514, 3026, 1924, 2812, 1753, 2307, 2319, 2246, 2315, 2398, 2361, 1622, 2561, 1844, 2037, 1991, 2455, 2538, 2592, 1745, 1577, 2073, 2316, 2690, 1656, 1607, 1962, 2221, 2445, 2019, 1880, 2219, 2516, 2250, 2458, 1731, 2308,}, 
                               {2359, 2300, 1940, 2981, 2894, 2164, 2284, 1990, 2717, 2677, 2824, 2502, 2741, 2373, 3261, 2204, 2729, 2236, 3012, 2669, 3173, 2327, 2450, 2495, 2091, 2577, 2445, 2492, 2018, 2970, 2718, 2394, 1979, 2038, 2701, 2756, 2832, 1691, 1628, 2506, 2705, 2169, 2330, 2110, 2212, 2708, 2047, 2963, 2012, 3191,}, 
                               {2689, 2344, 2223, 2572, 2725, 2267, 2135, 1973, 3186, 3182, 2672, 2120, 3191, 2643, 2999, 2003, 2919, 2855, 2925, 2678, 3236, 2202, 2993, 2460, 2445, 2565, 2636, 2620, 2455, 2900, 2896, 2785, 1958, 2251, 2443, 2683, 3226, 1916, 2106, 2529, 2703, 2533, 2210, 2258, 2333, 2931, 2414, 2497, 1803, 3045,}, 
                               {2640, 2581, 2161, 2601, 2713, 1722, 2173, 1591, 2729, 2786, 2765, 2427, 2837, 2540, 2814, 2460, 2944, 2616, 2784, 2328, 2884, 2350, 2746, 2202, 2132, 2846, 2323, 2220, 2331, 2743, 2800, 3046, 1692, 2152, 2059, 2710, 2945, 2134, 1735, 2585, 2501, 2376, 1999, 2488, 2439, 2614, 1911, 2719, 1741, 2532,}, 
                               {2241, 2421, 2295, 2777, 2586, 2167, 2177, 1675, 3118, 2624, 2827, 2201, 2942, 2237, 2823, 2450, 2706, 2136, 2681, 2314, 2989, 1934, 2819, 2348, 2084, 2648, 2323, 2575, 2377, 2404, 2900, 2693, 1420, 2156, 2582, 2769, 2749, 1809, 1648, 2282, 2213, 2336, 1863, 2352, 2338, 2883, 2122, 2807, 1681, 3115,}, 
                               {2460, 2519, 2087, 2956, 3067, 1885, 2543, 2285, 3159, 2735, 3237, 2396, 3318, 3079, 3516, 2664, 3372, 2378, 2867, 2413, 3157, 2704, 3044, 2659, 2145, 2943, 2453, 2612, 2410, 2444, 2982, 2640, 2238, 2128, 2542, 2818, 2950, 2354, 1885, 2901, 2690, 2471, 2210, 2472, 2854, 2766, 2404, 3123, 1950, 2952,}, 
                               {2616, 2336, 1974, 2655, 3050, 2267, 2686, 2148, 2737, 2795, 2736, 2311, 3039, 2449, 2906, 2306, 2894, 2976, 2752, 2771, 3266, 2378, 2664, 2444, 2110, 2447, 2731, 2636, 1864, 3254, 2582, 2279, 2233, 2262, 2468, 2377, 3156, 1666, 1859, 2642, 3035, 2479, 2489, 2251, 2497, 2717, 2137, 2674, 1778, 3124,}, 
                               {2264, 2276, 2108, 2503, 2525, 2060, 2567, 2051, 2538, 2446, 2820, 2075, 3400, 2373, 2978, 2244, 2732, 2273, 2443, 2483, 2688, 2044, 2823, 2348, 1927, 2461, 2576, 2449, 2235, 2784, 2710, 2284, 1890, 2000, 2371, 2232, 2910, 1751, 1672, 2155, 2463, 2606, 2157, 2229, 2524, 2608, 2428, 2538, 1657, 2782,}, 
                               {1569, 1485, 1237, 1938, 2108, 1316, 1369, 1183, 1541, 1929, 1956, 1549, 1901, 1461, 1731, 1533, 1759, 1803, 1988, 1643, 2210, 1522, 1787, 1531, 982, 1618, 1297, 1468, 1347, 2056, 1834, 1711, 1305, 1558, 1915, 1736, 1797, 1065, 1220, 1869, 2003, 1542, 1379, 1076, 1325, 1835, 1557, 1604, 1231, 1998, },
                               {1961, 1568, 1462, 1884, 1948, 1878, 2072, 1816, 2168, 2041, 1771, 1613, 2141, 1650, 2510, 1818, 2539, 1778, 2219, 2027, 2418, 1828, 1749, 2055, 1860, 1941, 1736, 1941, 1361, 1973, 2297, 2029, 1694, 1660, 1661, 1917, 2476, 1312, 1447, 1864, 2222, 1893, 1855, 1898, 1877, 2365, 1758, 1755, 1333, 2324,}, 
                               {2122, 2086, 1870, 2308, 2190, 1908, 1881, 1481, 2559, 2431, 2278, 1814, 2478, 2115, 2620, 1807, 2796, 2294, 2887, 2342, 2845, 1989, 2263, 2189, 2020, 2448, 1986, 2308, 2012, 2607, 2838, 2651, 1837, 2070, 1883, 2333, 3190, 1589, 1674, 2312, 2327, 2149, 1976, 2112, 2034, 2549, 2064, 2260, 1803, 2587,}, 
                               {2454, 2112, 2011, 2413, 2446, 1899, 2101, 1716, 2618, 2511, 2528, 2000, 2885, 2490, 2570, 2031, 2720, 2247, 2495, 2200, 2672, 2122, 2766, 2252, 1879, 2322, 2290, 2448, 2089, 2586, 2699, 2298, 1627, 1722, 2074, 2216, 3039, 1629, 1596, 2103, 2336, 2484, 1916, 2151, 2218, 2547, 2155, 2340, 1617, 2581,}, 
                               {2263, 2538, 2047, 2792, 2687, 2121, 2540, 2109, 2810, 2639, 2594, 2202, 3014, 2807, 3223, 2311, 3188, 2582, 2896, 2512, 3212, 2232, 2609, 2542, 2384, 2844, 2474, 2856, 2396, 2700, 2720, 2614, 2266, 2211, 2389, 2866, 3177, 1939, 1835, 2749, 2589, 2361, 2121, 1904, 2613, 2774, 2228, 2775, 2124, 3023,}, 
                               {2196, 2352, 1797, 2392, 2531, 1853, 2012, 1636, 3151, 2479, 2691, 2150, 2546, 2462, 2631, 2326, 2779, 2383, 2365, 2214, 3014, 2039, 2692, 2294, 2048, 2567, 2222, 2325, 2113, 2035, 2660, 2462, 1549, 2059, 2283, 2681, 2645, 1814, 1578, 2417, 2142, 2145, 1652, 2074, 2347, 2770, 1795, 2704, 1416, 2785,}, };

    VEC2D_EXPECT_EQ(result, expected)
}