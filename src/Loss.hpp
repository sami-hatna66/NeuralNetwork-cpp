#ifndef Loss_hpp
#define Loss_hpp

#include "utils.hpp"

#include <numeric>

namespace Loss {

template <typename T>
class CategoricalCrossEntropy {
private:
    Vec2d<T> dInputs;
public:
    T calculate(Vec2d<T>& output, Vec2d<T>& y);
    std::vector<T> compute(Vec2d<T>& predictY, Vec2d<T>& actualY);
    void backward(Vec2d<T>& dValues, Vec2d<T>& actualY);
    Vec2d<T> getDInputs();
};

}

#endif
