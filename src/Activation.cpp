#include "Activation.h"

double* Activation::softmax(double* sums, int size) {
    double sum = 0.0;
    for (int i{0}; i < size; ++i) {
        sums[i] = exp(sums[i]);
        sum += sums[i];
    }
    for (int i{0}; i < size; ++i) {
        sums[i] /= sum;
    }
    return sums;
}

double* Activation::relu(double* sums, int size) {
    for (int i{0}; i < size; ++i) {
        if (sums[i] < 0) {
            sums[i] *= 0.01;
        } else if (sums[i] > 1) {
            sums[i] = 1.0 + 0.01 * (sums[i] - 1);
        }
    }
    return sums;
}

double* Activation::sigmoid(double* sums, int size) {
    for (int i{0}; i < size; ++i) {
        sums[i] = 1 / (1 + exp(-sums[i]));
    }
    return sums;
}
