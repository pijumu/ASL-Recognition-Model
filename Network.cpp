#include "Network.h"

Layer::Layer(std::string& act_func, int size, Matrix& weights):
    act_func(act_func),
    size(size),
    weights(weights),
    bias(bias), {}

Layer::Layer(std::string& act_func, int size, int row):
    act_func(act_func),
    size(size),
    weights(row, size),
    bias(new double[row]) {
    for (int i=0; i < row; ++i) {
        bias[i] = ((std::rand() % 100)) * 0.007 / row;
    }
}

Network::Network(const std::string& path, const std::string& train_or_predict) {
    if (train_or_predict == "predict") {
        *this.read_weights(const std::string& path);
    } else if (train_or_predict == "train") {
        *this.read_config(const std::string& path);
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

