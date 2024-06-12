#include "Network.h"
#include <iostream>

Network::Network(const std::string &path, const std::string &train_or_predict) {
    YAML::Node config = YAML::LoadFile("../" + path);
    size = config["network_size"].as<int>();
    layers.reserve(size);

    for (int index{0}; index < size; ++index) {
        YAML::Node layer_config = config["layers"][index];
        auto activate_function = layer_config["activate_function"].as<std::string>();

        if (train_or_predict == "predict") {
            auto **weights = new double *[layer_config["weights"].size()];
            for (int i{0}; i < layer_config["weights"].size(); ++i) {
                weights[i] = new double[layer_config["weights"][i].size()];
                for (int j{0}; j < layer_config["weights"][0].size(); ++j) {
                    weights[i][j] = layer_config["weights"][i][j].as<double>();
                }
            }
            Matrix matrix{
                static_cast<int>(layer_config["weights"].size()),
                static_cast<int>(layer_config["weights"][0].size()),
                weights
            };
            auto *bias = new double[layer_config["bias"].size()];
            for (int i{0}; i < layer_config["bias"].size(); ++i) {
                bias[i] = layer_config["bias"][i].as<double>();
            }
            int layer_size = static_cast<int>(layer_config["bias"].size());
            Layer layer{activate_function, layer_size, matrix, bias};
            layers.push_back(layer);
        } else if (train_or_predict == "train") {
            int size_of_initial_neurons = config["size_of_initial_neurons"].as<int>();
            int layer_size = layer_config["layer_size"].as<int>();
            int row;
            if (index == 0) {
                row = size_of_initial_neurons;
            } else {
                row = layers[index - 1].size;
            }

            Layer layer{activate_function, row, layer_size};
            layers.push_back(layer);
        }
        if (train_or_predict == "train") {
            dropout_prob = config["dropout_probability"].as<double>();
        } else {
            dropout_prob = 0.0;
        }
        if (index == size - 1 && activate_function == "softmax") {
            loss_func = CrossEntropy;
        } else if (index == size - 1 && activate_function != "softmax") {
            loss_func = MSE;
        }
    }
    dropout_ini_mask = new double[layers[0].weights.get_row()];
    for (int i = 0; i < layers[0].weights.get_row(); ++i) {
        dropout_ini_mask[i] = 1.0;
    }
}
