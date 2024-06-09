/**
 * @file Derivative.h
 * @brief This file contains the declaration of functions for calculating derivatives of activation functions.
 */

#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <iostream>
#include <cmath>
#include "Matrix.h"

namespace der {
    /**
     * @brief Calculate the derivative of the Rectified Linear Unit (ReLU) activation function.
     * @param neurons Pointer to an array containing the output values of the ReLU function.
     * @param size The size of the input array.
     * @return A pointer to an array containing the derivatives of the ReLU function.
     */
    double* relu(const double* neurons, int size);

    /**
     * @brief Calculate the derivative of the Sigmoid activation function.
     * @param neurons Pointer to an array containing the output values of the Sigmoid function.
     * @param size The size of the input array.
     * @return A pointer to an array containing the derivatives of the Sigmoid function.
     */
    double* sigmoid(const double* neurons, int size);

    /**
     * @brief Calculate the derivative of the Softmax activation function for a vector of neurons.
     * @param neurons Pointer to an array containing the output values of the Softmax function.
     * @param size The size of the input array.
     * @return A Matrix object containing the derivatives of the Softmax function.
     */
    Matrix softmax(const double* neurons, int size);
}

#endif