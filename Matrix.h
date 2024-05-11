#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
class Matrix {

  private:
    int row;
    int column;
    double** elements;

  public:
    Matrix(int row, int column);
    Matrix(int row, int column, double** elements);

    friend double* sum_vector(double* vector_1, const double* vector_2, int size);
    friend double* operator* (const Matrix& matrix, const double* vector);
    friend double* operator* (const double* vector, const Matrix& matrix);
    friend Matrix operator* (const Matrix& matrix1, const Matrix& matrix2);

    int get_row() const;
    int get_column() const;
    double element(int i, int j) const;
};
#endif