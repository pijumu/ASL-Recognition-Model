#ifndef TEST_FUNC_H
#define TEST_FUNC_H
#include "Matrix.h"
#include "Derivative.h"
#include "doctest.h"
#include "Activation.h"
bool operator==(const Matrix& matrix1, const Matrix& matrix2);
bool check_equal_vectors(const double* vector_1, const double* vector_2, int size);
#endif