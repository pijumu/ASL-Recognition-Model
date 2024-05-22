#include "Matrix.h"
#include <string>
#include <vector>


class Layer {
    // activation_function - название функции активации
    // size - кол-во нейронов в слою
    // weights - веса
    // bias - он самый
    // neurons - значение нейронов
    std::string act_func;
    int size;
    Matrix weights;
    double* bias_weights;
    double* neurons;
    double* neurons_err;
    double* sums; // for relu

  public:
    Layer(std::string& act_func, int size, Matrix& weights, double* bias);
    Laeyr(std::string& act_func, int size, int row);

};

class Network {

    int size;
    double** neurons_err;
    std::vector<Layer> layers;
    std::string loss_func;

  public:    
    Network(const std::string& path, const std::string& train_or_predict);
    
    void set_input();
    void forward_feed(double* initial_neurons);
    void back_propagation(double* initial_neurons);

    int predict();

    void read_config(const std::string& path);
    void read_weights(const std::string& path);
    void write_weights(const std::string& path);
};
