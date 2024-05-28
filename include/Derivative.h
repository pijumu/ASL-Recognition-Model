#ifndef DERIVATIVE_H
#define DERIVATIVE_H
#include <iostream>
#include <cmath>
#include "Matrix.h"

namespace der {
    double* relu(const double* sums, int size);
    double* sigmoid(const double* neurons, int size);
    Matrix softmax(const double* neurons, int size);
}
#endif
