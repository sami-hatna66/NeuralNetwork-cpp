#ifndef Loss_hpp
#define Loss_hpp

#include "utils.hpp"

#include <numeric>

namespace Loss {

class CategoricalCrossEntropy {
private:
    DoubleVec2d dInputs;
public:
    double calculate(DoubleVec2d& output, DoubleVec2d& y);
    std::vector<double> compute(DoubleVec2d& predictY, DoubleVec2d& actualY);
    void backward(DoubleVec2d& dValues, DoubleVec2d& actualY);
    DoubleVec2d getDInputs();
};

}

#endif
