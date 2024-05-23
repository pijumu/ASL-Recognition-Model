#include "Matrix.h"
#include "Derivative.h"
#include <string>
#include <vector>
#include "Activation.h"
#include "yaml-cpp/yaml.h"
#include <fstream>
class Layer {
    // activation_function - название функции активации
    // size - кол-во нейронов в слою
    // weights - веса
    // bias - он самый
    // neurons - значение нейронов
  public:
    std::string act_func;
    int size;
    Matrix weights;
    double* bias_weights;
    double* neurons;
    double* neurons_err;
    double* sums; // for relu

    Layer(std::string& act_func, int size, Matrix& weights, double* bias);
    Layer(std::string& act_func, int size, int row);

};

class Network {
  public:
    int size;
    std::string loss_func;
    std::vector<Layer> layers;
      
    Network(const std::string& path, const std::string& train_or_predict);
    
    void set_input();
    void forward_feed(double* initial_neurons);
    void back_propagation(double* initial_neurons);

    int predict();

    void write_weights(const std::string& path);
};
