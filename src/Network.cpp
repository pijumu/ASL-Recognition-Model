#include "Network.h"

Layer::Layer(const std::string& act_func, int size, Matrix& weights, double* bias_weights):
    act_func(act_func),
    size(size),
    weights(weights),
    gradient(weights.get_row(), size, 0.0),
    bias_weights(bias_weights),
    bias_gradient(new double[size]{})
{}

Layer::Layer(std::string& act_func, int size, int row):
    act_func(act_func),
    size(size),
    weights(row, size),
    gradient(row, size, 0.0),
    bias_weights(new double[size]),
    bias_gradient(new double[size]{})
{
    for (int i=0; i < size; ++i) {
        bias_weights[i] = ((std::rand() % 100)) * 0.007 / size;
    }
}

void Network::forward_feed()
{
    for (int i=0; i < size; ++i) {
        if (i == 0) {
            if (layers[i].act_func == "relu") {
                layers[i].sums = Matrix::sum_vector(
                        initial_neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size
                );
                layers[i].neurons = act::relu(
                        layers[i].sums,
                        layers[i].size
                ); 
            } 
            else if (layers[i].act_func == "sigmoid") {
                layers[i].neurons = act::sigmoid(
                        Matrix::sum_vector(
                                initial_neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            } else {
                layers[i].neurons = act::softmax(
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
                layers[i].neurons = act::relu(
                        layers[i].sums,
                        layers[i].size
                ); 
            } 
            else if (layers[i].act_func == "sigmoid") {
                layers[i].neurons = act::sigmoid(
                        Matrix::sum_vector(
                                layers[i-1].neurons * layers[i].weights,
                                layers[i].bias_weights,
                                layers[i].size
                        ),
                        layers[i].size
                );
            } else {
                layers[i].neurons = act::softmax(
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

void Network::back_propagation(double* expected)
{
    layers[size - 1].de_ds = new double[layers[size - 1].size];
    if (loss_func == CrossEntropy) {
        for (int i{0}; i < layers[size - 1].size; ++i) {
            layers[size - 1].de_ds[i] = layers[size - 1].neurons[i] - expected[i];
        }
    } else if (loss_func == MSE){
        double* der;
        if (layers[size - 1].act_func == "relu"){
            der = der::relu(
                layers[size - 1].sums,
                layers[size - 1].size
            );
        } else if (layers[size - 1].act_func == "sigmoid") {
            der = der::sigmoid(
                layers[size - 1].neurons,
                layers[size - 1].size
            );
        }
        for (int i{0}; i < layers[size - 1].size; ++i) {
            layers[size - 1].de_ds[i] = 2 * (layers[size - 1].neurons[i] - expected[i]) * der[i];
        }
    }
    for (int i{size - 2}; i >= 0; --i) {
        if (layers[i].act_func == "relu") {
            layers[i].de_ds = Matrix::multy_elements(
                layers[i + 1].weights * layers[i + 1].de_ds,
                der::relu(
                    layers[i].sums,
                    layers[i].size
                ),
                layers[i].size
            );
        } else if (layers[i].act_func == "sigmoid") {
            layers[i].de_ds = Matrix::multy_elements(
                layers[i + 1].weights * layers[i + 1].de_ds,
                der::sigmoid(
                    layers[i].neurons,
                    layers[i].size
                ),
                layers[i].size
            );
        }
        else if (layers[i].act_func == "softmax"){
            layers[i].de_ds = layers[i + 1].weights *
            layers[i + 1].de_ds *
            der::softmax(
                layers[i].neurons, 
                layers[i].size
            );
        }
    }
    for (int j=0; j<layers[0].gradient.get_row(); ++j) {
        for (int t=0; t<layers[0].gradient.get_column(); ++t) {
            layers[0].gradient.elem(j, t) -= initial_neurons[j] * layers[0].de_ds[t];
        }
    }
    for (int t=0; t<layers[0].gradient.get_column(); ++t) {
        layers[0].bias_gradient[t] -= layers[0].de_ds[t];
    }

    for (int i=1; i<size; ++i) {
        for (int j=0; j<layers[i].gradient.get_row(); ++j) {
            for (int t=0; t<layers[i].gradient.get_column(); ++t) {
                layers[i].gradient.elem(j, t) -= layers[i-1].neurons[j] * layers[i].de_ds[t];
            }
        }
        for (int t=0; t<layers[i].gradient.get_column(); ++t) {
            layers[i].bias_gradient[t] -= layers[i].de_ds[t];
        }
    }
}

void Network::set_input(double* initial_neurons)
{
    this->initial_neurons = initial_neurons;
}

void Network::update_weights(const double lr)
{
    for (int i{0}; i < size; ++i) {
        for (int x{0}; x<layers[i].gradient.get_row(); ++x) {
            for (int y{0}; y<layers[i].gradient.get_column(); ++y) {
                layers[i].weights.elem(x, y) += lr * layers[i].gradient.elem(x, y);
                layers[i].gradient.elem(x, y) = 0.0;
            }
        }
        for (int t{0}; t<layers[i].gradient.get_column(); ++t) {
            layers[i].bias_weights[t] += lr * layers[i].bias_gradient[t];
            layers[i].bias_gradient[t] = 0.0;
        }
    }
}

int Network::predict()
{
    int index = -1;
    double max = 0.0;
    for (int i=0; i<layers[size-1].size; ++i) {
        if (layers[size-1].neurons[i] >= max) {
            index = i;
            max = layers[size-1].neurons[i];
        }
    }
    return index;
}
