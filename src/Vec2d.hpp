#ifndef Vec2d_hpp
#define Vec2d_hpp

#include <vector>
#include <span>

template <typename T>
concept DecimalType = std::is_floating_point_v<T>;

template <DecimalType T>
class Vec2d {
private:
    std::vector<T> data;
    int rows;
    int cols;

public:
    Vec2d();
    Vec2d(const std::vector<std::vector<T>>& newData);
    Vec2d(const std::vector<std::vector<T>>&& newData);
    Vec2d(const int pRows, const int pCols);
    Vec2d(const int pRows, const int pCols, const T initVal);
    Vec2d(const Vec2d<T>& other);

    Vec2d<T>& operator=(const std::vector<std::vector<T>>& newData);
    Vec2d<T>& operator=(const std::vector<std::vector<T>>&& newData);
    Vec2d<T>& operator=(const Vec2d<T>& other);
    Vec2d<T>& operator=(const Vec2d<T>&& other);
    std::span<T> operator[](int idx);
    std::vector<T> operator[](int idx) const;

    void push_back(std::vector<T>& newRow);
    void push_back(std::vector<T>&& newRow);
    void push_back(std::span<T>& newRow);
    void push_back(std::span<T>&& newRow);
    void clear();
    void resize(int newRows, int newCols);

    void addToRow(int rowNum, T val);

    const std::vector<T>& getData() const;
    int getRows() const;
    int getCols() const;
    int size() const;
};

// New approach:
//  - replace all range-based for loops with normal for loops
//  - replace push_back on span with custom functions insertInRow(Vec2d, rowNum, vector) and insertInRow(Vec2d, rowNum, T)

#endif
