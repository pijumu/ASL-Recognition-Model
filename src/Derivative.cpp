#include "Derivative.h"

namespace der {
    double *relu(const double *neurons, int size) {
        auto *relu_neurons = new double[size];
        for (int i{0}; i < size; ++i) {
            if (neurons[i] > 0) {
                relu_neurons[i] = 0.1;
            } else {
                relu_neurons[i] = 0.0;
            }
        }
        return relu_neurons;
    }

    double *sigmoid(const double *neurons, int size) {
        auto *sigmoid_neurons = new double[size];
        for (int i{0}; i < size; ++i) {
            sigmoid_neurons[i] = neurons[i] * (1 - neurons[i]);
        }
        return sigmoid_neurons;
    }

    Matrix softmax(const double *neurons, int size) {
        auto **softmax_neurons = new double *[size];
        for (int i{0}; i < size; ++i) {
            softmax_neurons[i] = new double[size];
            for (int j{0}; j < size; ++j) {
                if (i == j) {
                    softmax_neurons[i][j] = neurons[i] * (1 - neurons[i]);
                } else {
                    softmax_neurons[i][j] = -neurons[i] * neurons[j];
                }
            }
        }
        return Matrix{size, size, softmax_neurons};
    }
}
