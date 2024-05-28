#include "Derivative.h"

namespace der {
    double* relu(const double* sums, int size){
        auto* relu_neurons = new double[size];
        for (int i{0}; i < size; ++i) {
            if (0 < sums[i] and sums[i] > 1) {
                relu_neurons[i] = 0.01;
            } else {
                relu_neurons[i] = 1;
            }
        }
        return relu_neurons;
    }

    double* sigmoid(const double* neurons, int size){
        auto* sigmoid_neurons = new double[size];
        for (int i{0}; i < size; ++i){
            sigmoid_neurons[i] = pow(M_E, -neurons[i])/pow(1 + pow(M_E, -neurons[i]), 2);
        }
        return sigmoid_neurons;
    }

    Matrix softmax(const double* neurons, int size){
        auto** softmax_neurons = new double*[size];
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
        return Matrix {size, size, softmax_neurons};
    }
}
