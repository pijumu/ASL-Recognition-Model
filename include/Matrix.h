/**
 * @file Matrix.h
 * @brief This file contains the declaration of the Matrix class and related functions for matrix operations.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

/**
 * @class Matrix
 * @brief A class to represent a matrix and perform matrix operations.
 */
class Matrix {
public:
 /**
  * @brief Constructor for creating a matrix with specified number of rows and columns and with random values.
  * @param row The number of rows in the matrix.
  * @param column The number of columns in the matrix.
  */
 Matrix(int row, int column);

 /**
  * @brief Constructor for creating a matrix with specified elements.
  * @param row The number of rows in the matrix.
  * @param column The number of columns in the matrix.
  * @param elements A 2D array containing the elements of the matrix.
  */
 Matrix(int row, int column, double** elements);

 /**
  * @brief Constructor for creating a matrix with all elements set to a specified value.
  * @param row The number of rows in the matrix.
  * @param column The number of columns in the matrix.
  * @param elem The value to set for all elements of the matrix.
  */
 Matrix(int row, int column, double elem);

 /**
  * @brief Multiplies each element of two vectors element-wise.
  * @param vector_1 The first vector.
  * @param vector_2 The second vector.
  * @param size The size of the vectors.
  * @return A pointer to an array containing the result of element-wise multiplication.
  */
 static double* multy_elements(double* vector_1, const double* vector_2, int size);

 /**
  * @brief Adds two vectors element-wise.
  * @param vector_1 The first vector.
  * @param vector_2 The second vector.
  * @param size The size of the vectors.
  * @return A pointer to an array containing the result of element-wise addition.
  */
 static double* sum_vector(double* vector_1, const double* vector_2, int size);

 /**
  * @return The number of rows in the matrix.
  */
 [[nodiscard]] int get_row() const;

 /**
  * @return The number of columns in the matrix.
  */
 [[nodiscard]] int get_column() const;

 /**
  * @brief Get the value of a specific element in the matrix.
  * @param i The row index of the element.
  * @param j The column index of the element.
  * @return The value of the element at the specified position.
  */
 [[nodiscard]] double element(int i, int j) const;

 /**
  * @brief Get a reference to a specific element in the matrix for modification.
  * @param i The row index of the element.
  * @param j The column index of the element.
  * @return A reference to the element at the specified position.
  */
 double& elem(int i, int j);

private:
 int row; /**< The number of rows in the matrix. */
 int column; /**< The number of columns in the matrix. */
 double** elements; /**< A 2D array containing the elements of the matrix. */
};

/**
 * @brief Overloaded operator for multiplying a matrix by a vector.
 * @param matrix The matrix to be multiplied.
 * @param vector The vector to be multiplied.
 * @return A pointer to an array containing the result of the multiplication.
 */
double* operator* (const Matrix& matrix, const double* vector);

/**
 * @brief Overloaded operator for multiplying a vector by a matrix.
 * @param vector The vector to be multiplied.
 * @param matrix The matrix to be multiplied.
 * @return A pointer to an array containing the result of the multiplication.
 */
double* operator* (const double* vector, const Matrix& matrix);

/**
 * @brief Overloaded operator for multiplying two matrices.
 * @param matrix1 The first matrix to be multiplied.
 * @param matrix2 The second matrix to be multiplied.
 * @return A new Matrix object containing the result of the multiplication.
 */
Matrix operator* (const Matrix& matrix1, const Matrix& matrix2);

#endif