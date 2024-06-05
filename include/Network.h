#include "Matrix.h"
#include "Derivative.h"
#include <string>
#include <vector>
#include "Activation.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

enum LossFunc {
    CrossEntropy,
    MSE,
};

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
    double* de_ds;
    double* sums; // for relu

    Layer(std::string& act_func, int size, Matrix& weights, double* bias);
    Layer(std::string& act_func, int size, int row);

};

class Network {
  public:
    int size;
    LossFunc loss_func;
    std::vector<Layer> layers;
    double* initial_neurons;
      
    Network(const std::string& path, const std::string& train_or_predict);
    
    void set_input(double* initial_neurons);
    void forward_feed();
    void back_propagation(double* expected);
    void update_weights(double lr);

    int predict();

    void write_weights(const std::string& path);
};
