#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
class Matrix {
    int row;
    int column;
    double **elements;
    public:
        void init(int row, int column);
        static void sum_vector(double* vector_1, const double* vector_2, int size);
        static void multiply(Matrix& matrix, const double* vector, int size, double* result);
        int get_row();
        int get_column();
        double element(int i, int j);
};
#endif