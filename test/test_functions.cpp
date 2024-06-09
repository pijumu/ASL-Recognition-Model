#include "include/test_functions.h"
bool operator==(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.get_column() != matrix2.get_column() || matrix1.get_row() != matrix2.get_row()) {
        return false;
    }
    for (int i{0}; i < matrix1.get_row(); ++i) {
        for (int j{0}; j < matrix1.get_column(); ++j) {
            if (matrix1.element(i, j) - matrix2.element(i, j) > 0.00000001) {
                return false;
            }
        }
    }
    return true;
}

bool check_equal_vectors(const double* vector_1, const double* vector_2, int size) {
    for (int i{0}; i < size; ++i) {
        if (vector_1[i] - vector_2[i] > 0.000001) {
            return false;
        }
    }
    return true;
}