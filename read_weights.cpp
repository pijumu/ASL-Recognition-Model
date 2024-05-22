#include "Network.h"

void Network::read_weights(const std::string& path){
    YAML::Node config = YAML::LoadFile("../" + path);

    size = config["network size"].as<int>();
    layers.reserve(size);
    loss_function = config["loss function"].as<std::string>();

    for (int index{0}; index < size; ++index) {
        YAML::Node layer_config = config["layers"][index];

        std::string activate_function = layer_config["activate function"].as<std::string>();
    
        double** weights = new double* [layer_config["weights"].size()];
        for (int i{0}; i < layer_config["weights"].size(); ++i) {
            weights[i] = new double[layer_config["weights"].size()];
            for (int j{0}; j < layer_config["weights"][0].size(); ++j) {
                weights[i][j] = layer_config["weights"][i][j].as<double>();
            }
        }
        Matrix matrix(layer_config["weights"].size(), layer_config["weights"][0].size(), weights);
        
        double* bias = new double[layer_config["bias"].size()];
        for (int i{0}; i < layer_config["bias"].size(); ++i) {
            bias[i] = layer_config["bias"][i].as<double>();
        }
        
        int layer_size = layer_config["neurons"].size();
        Layer layer = Layer{activate_function, layer_size, matrix, bias};
        
        layers.push_back(layer);

    }
};