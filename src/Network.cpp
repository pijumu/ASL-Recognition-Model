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

Network::forward_feed(double* initial_neurons) {
    for (int i=0; i < size; ++i) {
        if (i == 0) {
            if (layers[i].act_func == "relu") {
                layers[i].sums = sum_vector(
                        initial_neurons * layers[i].weights,
                        bias,
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
                        bias,
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
}
