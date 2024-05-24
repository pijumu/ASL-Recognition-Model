#include "Network.h"

Layer::Layer(std::string& act_func, int size, Matrix& weights, double* bias_weights):
    act_func(act_func),
    size(size),
    weights(weights),
    bias_weights(bias_weights) {};

Layer::Layer(std::string& act_func, int size, int row):
    act_func(act_func),
    size(size),
    weights(row, size),
    bias_weights(new double[row]) {
    for (int i=0; i < row; ++i) {
        bias_weights[i] = ((std::rand() % 100)) * 0.007 / row;
    }
}
/*
Network::forward_feed(double* initial_neurons) {
    for (int i=0; i < size; ++i) {
        if (i == 0) {
            if (layers[i].act_func == "relu") {
                layers[i].sums = sum_vector(
                        initial_neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size,
                );
                layers[i].neurons = Activation::relu(
                        layers[i].sums,
                        layers[i].size
                ); 
            } 
            layers[i].neurons = Activation::r(
                    sum_vector(
                        initial_neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size,
                    )
             );
        } else {
            layers[i].neurons = sum_vector(
                    layers[i-1].neurons * layers[i].weights,
                    bias,
                    layers[i].size,
            );
        }
    }
}*/

void Network::forward_feed(double* initial_neurons) {
    for (int i=0; i < size; ++i) {
        if (i == 0) {
            if (layers[i].act_func == "relu") {
                layers[i].sums = Matrix::sum_vector(
                        initial_neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size
                );
                layers[i].neurons = Activation::relu(
                        layers[i].sums,
                        layers[i].size
                ); 
            } 
            else if (layers[i].act_func == "sigmoid") {
                layers[i].neurons = Activation::sigmoid(
                        Matrix::sum_vector(
                                initial_neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            } else {
                layers[i].neurons = Activation::softmax(
                        Matrix::sum_vector(
                                initial_neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            }
        } else {
            if (layers[i].act_func == "relu") {
                layers[i].sums = Matrix::sum_vector(
                        layers[i-1].neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size
                );
                layers[i].neurons = Activation::relu(
                        layers[i].sums,
                        layers[i].size
                ); 
            } 
            else if (layers[i].act_func == "sigmoid") {
                layers[i].neurons = Activation::sigmoid(
                        Matrix::sum_vector(
                                layers[i-1].neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            } else {
                layers[i].neurons = Activation::softmax(
                        Matrix::sum_vector(
                                layers[i-1].neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            }
        }
    }
}

void Network::back_propagation(double* result) {
    layers[size - 1].neurons_err = new double[layers[size - 1].size];
    if (loss_func == "crossentropy") {
        for (int i{0}; i < layers[size - 1].size; ++i) {
            layers[size - 1].neurons_err[i] = layers[size - 1].neurons[i] - result[i];
        }
    } else if (loss_func == "mse"){
        double* der;
        if (layers[size - 1].act_func == "relu"){
            der = Derivative::relu(
                layers[size - 1].sums,
                layers[size - 1].size
            );
        } else if (layers[size - 1].act_func == "sigmoid") {
            der = Derivative::sigmoid(
                layers[size - 1].neurons,
                layers[size - 1].size
            );
        }
        for (int i{0}; i < layers[size - 1].size; ++i) {
            layers[size - 1].neurons_err[i] = 2 * (layers[size - 1].neurons[i] - result[i]) * der[i];
        }
    }
    for (int i{size - 2}; i >= 0; --i) {
        if (layers[i].act_func == "relu") {
            layers[i].neurons_err = Matrix::multy_elements(
                layers[i + 1].weights * layers[i + 1].neurons_err,
                Derivative::relu(
                    layers[i].sums,
                    layers[i].size
                ),
                layers[i].size
            );
        } else if (layers[i].act_func == "sigmoid") {
            layers[i].neurons_err = Matrix::multy_elements(
                layers[i + 1].weights * layers[i + 1].neurons_err,
                Derivative::sigmoid(
                    layers[i].neurons,
                    layers[i].size
                ),
                layers[i].size
            );
        }
        else if (layers[i].act_func == "softmax"){
            layers[i].neurons_err = layers[i + 1].weights *
            layers[i + 1].neurons_err *
            Derivative::softmax(
                layers[i].neurons, 
                layers[i].size
            );
        }
    }
}
