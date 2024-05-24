#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <iostream>
#include <cmath>

class Activation {
  public:
    static double* softmax(double *sums, int size);
    static double* relu(double *sums, int size);
    static double* sigmoid(double *sums, int size);
};
#endif