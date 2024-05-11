#ifndef DERIVATIVE_H
#define DERIVATIVE_H
#include <iostream>
#include <cmath>
class Derivative {
    public:
        static double* relu(double* neurons, int size);
        static double* sigmoid(double* neurons, int size);
        static double** softmax(double* neurons, int size);
        //static double dropout(double neuron);
};
#endif