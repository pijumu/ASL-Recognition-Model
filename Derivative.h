#ifndef DERIVATIVE_H
#define DERIVATIVE_H
#include <iostream>
#include <cmath>
#include "Matrix.h"

class Derivative {
  public:
    static double* relu(const double* sums, int size);
    static double* sigmoid(const double* neurons, int size);
    static Matrix softmax(const double* neurons, int size);
};
#endif