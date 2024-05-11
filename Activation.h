#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <iostream>
#include <cmath>

class Activation {
  public:
    static void softmax(double *sums, int size);
    static void relu(double *sums, int size);
    static void sigmoid(double *sums, int size);
};
#endif