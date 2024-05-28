#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <iostream>
#include <cmath>

namespace act {
    double* softmax(double *sums, int size);
    double* relu(double *sums, int size);
    double* sigmoid(double *sums, int size);
}
#endif
