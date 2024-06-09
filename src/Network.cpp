#include "Network.h"
#include <random>

Layer::Layer(const std::string& act_func, int size, Matrix& weights, double* bias_weights):
    act_func(act_func),
    size(size),
    weights(weights),
    gradient(weights.get_row(), size, 0.0),
    bias_weights(bias_weights),
    bias_gradient(new double[size]{}),
    dropout_mask(new double[size])
{
    for (int i=0; i<size;++i) {
        dropout_mask[i] = 1.0;
    }
}

Layer::Layer(std::string& act_func, int row, int size):
    act_func(act_func),
    size(size),
    weights(row, size),
    gradient(row, size, 0.0),
    bias_weights(new double[size]),
    bias_gradient(new double[size]{}),
    dropout_mask(new double[size])
{
    for (int i=0; i < size; ++i) {
        bias_weights[i] = ((std::rand() % 10)) * 0.0007 / size;
        dropout_mask[i] = 1.0;

    }
}

void Network::forward_feed()
{
    initial_neurons = Matrix::multy_elements(
                initial_neurons,
                dropout_ini_mask,
                layers[0].weights.get_row()
    );
    for (int i=0; i < size; ++i) {
        if (i == 0) {
            if (layers[i].act_func == "relu") {
                layers[i].neurons = act::relu(
                    Matrix::sum_vector(
                        initial_neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size
                    ),
                        layers[i].size
                ); 
            } else if (layers[i].act_func == "sigmoid") {
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
                layers[i].neurons = act::relu(
                    Matrix::sum_vector(
                        layers[i-1].neurons * layers[i].weights,
                        layers[i].bias_weights,
                        layers[i].size
                    ),
                    layers[i].size
                ); 
            } else if (layers[i].act_func == "sigmoid") {
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
        if (i < size-1) {
            layers[i].neurons = Matrix::multy_elements(
                layers[i].neurons,
                layers[i].dropout_mask,
                layers[i].size
            );
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
                layers[size - 1].neurons,
                layers[size - 1].size
            );
        } else if (layers[size - 1].act_func == "sigmoid") {
            der = der::sigmoid(
                layers[size - 1].neurons,
                layers[size - 1].size
            );
        }
        for (int i{0}; i < layers[size - 1].size; ++i) {
            layers[size - 1].de_ds[i] = 2 * (layers[size - 1].neurons[i] - expected[i]) * der[i] / static_cast<double>(size);
        }
    }
    for (int i{size - 2}; i >= 0; --i) {
        if (layers[i].act_func == "relu") {
            layers[i].de_ds = Matrix::multy_elements(
                layers[i + 1].weights * layers[i + 1].de_ds,
                der::relu(
                    layers[i].neurons,
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
        for (int k=0; k<layers[i].gradient.get_column(); ++k) {
            layers[i].bias_gradient[k] -= layers[i].de_ds[k];
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

void Network::dropout_mask(double p) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i=0; i<size-1 ; ++i) {
        for (int j=0; j<layers[i].size; ++j) {
            if (const double random_value = dis(gen); random_value <= p) {
                layers[i].dropout_mask[j] = 0.0;
            } else {
                layers[i].dropout_mask[j] = 1.0;
            }
        }
    }
    dropout_ini_mask = new double[layers[0].weights.get_row()];
    for (int x=0; x < layers[0].weights.get_row(); ++x) {
        if (const double random_value = dis(gen); random_value <= p) {
            dropout_ini_mask[x] = 0.0;
        } else {
            dropout_ini_mask[x] = 1.0;
        }
    }
}