#include "Activation.h"

namespace act {
    double* softmax(double* sums, int size) {
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

    double* relu(double* sums, int size) {
        for (int i{0}; i < size; ++i) {
            sums[i] = std::max(0.0, 0.1*sums[i]);
        }
        return sums;
    }

    double* sigmoid(double* sums, int size) {
        for (int i{0}; i < size; ++i) {
            sums[i] = 1 / (1 + exp(-sums[i]));
        }
        return sums;
    }
}
