/**
 * @file Activation.h
 * @brief This file contains declarations of activation functions for neural networks.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <cmath>

namespace act {
    /**
     * @brief Applies the softmax activation function to an array of sums.
     * @param sums An array of sums.
     * @param size The size of the array.
     * @return A pointer to an array of doubles containing the softmax values.
     */
    double *softmax(double *sums, int size);

    /**
     * @brief Applies the ReLU (Rectified Linear Unit) activation function to an array of sums.
     * @param sums An array of sums.
     * @param size The size of the array.
     * @return A pointer to an array of doubles containing the ReLU values.
     */
    double *relu(double *sums, int size);

    /**
     * @brief Applies the sigmoid activation function to an array of sums.
     * @param sums An array of sums.
     * @param size The size of the array.
     * @return A pointer to an array of doubles containing the sigmoid values.
     */
    double *sigmoid(double *sums, int size);
}

#endif
