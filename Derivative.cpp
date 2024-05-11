#include "derivative.h"
double* Derivative::relu(double* neurons, int size){
    double* relu_neurons = new double[size];
    for (int i{0}; i < size; ++i) {
        if (neurons[i] < 0 or neurons[i] > 1) {
            relu_neurons[i] = 0.01;
        }
        else {
            relu_neurons[i] = 1;
        }
    }
    return relu_neurons;
}

double* Derivative::sigmoid(double* neurons, int size){
    double* sigmoid_neurons = new double[size];
    for (int i{0}; i < size; ++i){
        sigmoid_neurons[i] = pow(M_E, -neurons[i])/pow(1 + pow(M_E, -neurons[i]), 2);
    }
    return sigmoid_neurons;
}

double** Derivative::softmax(double* neurons, int size){
    double** softmax_neurons = new double*[size];
    for (int i{0}; i < size; ++i) {
        softmax_neurons[i] = new double[size];
        for (int j{0}; j < size; ++j) {
            if (i == j) {
                softmax_neurons[i][j] = neurons[i] * (1 - neurons[i]);
            }
            else {
                softmax_neurons[i][j] = -neurons[i] * neurons[j];
            }
        }
    }

    return softmax_neurons;
}
