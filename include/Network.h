/**
 * @file Network.h
 * @brief This file contains the declaration of classes for implementing neural network layers and network operations.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "Matrix.h"
#include "Derivative.h"
#include <string>
#include <vector>
#include "Activation.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

/**
 * @enum LossFunc
 * @brief Enum representing different loss functions for training the neural network.
 */
enum LossFunc {
 CrossEntropy, /**< Cross-entropy loss function. */
 MSE, /**< Mean squared error loss function. */
};

/**
 * @class Layer
 * @brief Class representing a layer in a neural network.
 */
class Layer {
public:
 std::string act_func; /**< Activation function used in the layer. */
 int size; /**< Number of neurons in the layer. */
 Matrix weights; /**< Matrix of weights connecting the neurons in this layer to the next layer. */
 Matrix gradient; /**< Matrix of gradients for weight updates. */
 double* bias_weights; /**< Array of bias weights for each neuron in the layer. */
 double* bias_gradient; /**< Array of gradients for bias weight updates. */
 double* neurons; /**< Array of output values from the neurons in the layer. */
 double* de_ds; /**< Array of derivatives of the activation function with respect to the summed inputs. */

 /**
  * @brief Constructor for Layer class with specified activation function, size, weights, and bias.
  * @param act_func The activation function for the layer.
  * @param size The number of neurons in the layer.
  * @param weights Matrix of weights connecting neurons in this layer to the next layer.
  * @param bias Array of bias weights for each neuron in the layer.
  */
 Layer(const std::string& act_func, int size, Matrix& weights, double* bias);

 /**
  * @brief Constructor for Layer class with specified activation function, size, and number of input rows.
  * @param act_func The activation function for the layer.
  * @param size The number of neurons in the layer.
  * @param row Number of rows for input weights matrix.
  */
 Layer(std::string& act_func, int row, int size);
};

/**
 * @class Network
 * @brief Class representing a neural network with multiple layers.
 */
class Network {
public:
 int size; /**< Number of layers in the network. */
 LossFunc loss_func; /**< Loss function used for training the network. */
 std::vector<Layer> layers; /**< Vector containing all layers in the network. */
 double* initial_neurons; /**< Array of initial input values to the network. */

 /**
  * @brief Constructor for Network class that loads network configuration from a YAML file.
  * @param path Path to the YAML configuration file.
  * @param train_or_predict Flag indicating whether to train or predict using the network.
  */
 Network(const std::string& path, const std::string& train_or_predict);

 /**
  * @brief Set initial input values to the network.
  * @param initial_neurons Array of initial input values.
  */
 void set_input(double* initial_neurons);

 /**
  * @brief Perform forward propagation through the network.
  */
 void forward_feed();

 /**
  * @brief Perform backpropagation to update weights based on expected output.
  * @param expected Array of expected output values.
  */
 void back_propagation(double* expected);

 /**
  * @brief Update weights in the network using a specified learning rate.
  * @param lr Learning rate for weight updates.
  */
 void update_weights(double lr);

 /**
  * @brief Make a prediction using the current network configuration.
  * @return Predicted output value.
  */
 int predict();

 /**
  * @brief Write updated weights to a specified file path.
  * @param path Path to write the updated weights.
  */
 void write_weights(const std::string& path);
};
#endif