#include "Matrix.h"

// Конструкторы
Matrix::Matrix(const int row, const int column): row(row), column(column), elements(new double* [row])
{
    for (int i{0}; i < row; ++i) {
        elements[i] = new double [column];
        for (int j{0}; j < column; ++j) {
            elements[i][j] = ((std::rand()%100)*0.0007);
        }
    }
}

Matrix::Matrix(const int row, const int column, double** elements): row(row), column(column), elements(elements)
{}

Matrix::Matrix(const int row, const int column, const double elem): row(row), column(column), elements(new double* [row])
{
    for (int i{0}; i < row; ++i) {
        elements[i] = new double [column];
        for (int j{0}; j < column; ++j) {
            elements[i][j] = elem; // NOLINT(*-msc50-cpp)
        }
    }
}

// Работа с полями
int Matrix::get_row() const
{
    return Matrix::row;
}

int Matrix::get_column() const
{
    return Matrix::column;
}

double Matrix::element(int i, int j) const
{
    return Matrix::elements[i][j];
}

double& Matrix::elem(int i, int j)
{
    return Matrix::elements[i][j];
}

// Работа с векторами
double* Matrix::sum_vector(double* vector_1, const double* vector_2, int size)
{
    for (int i{0}; i < size; ++i) {
        vector_1[i] += vector_2[i];
    }
    return vector_1;
}

double* Matrix::multy_elements(double* vector_1, const double* vector_2, int size)
{
    for (int i{0}; i < size; ++i) {
        vector_1[i] = vector_1[i] * vector_2[i];
    }
    return vector_1;
}

//Работа с матрицами
Matrix operator*(const Matrix& matrix1, const Matrix& matrix2)
{
    auto** elems = new double* [matrix1.get_row()];
    for (int i{0}; i < matrix1.get_row(); ++i) {
        elems[i] = new double [matrix2.get_column()];
    }
    for (int i{0}; i < matrix1.get_row(); ++i) {
        for (int j{0}; j < matrix2.get_column(); ++j) {
            elems[i][j] = 0;
            for (int k{0}; k < matrix1.get_column(); k++) {
                elems[i][j] += matrix1.element(i, k) * matrix2.element(k, j);
            }
        }
    }
    return Matrix {matrix1.get_row(), matrix2.get_column(), elems};
}

double* operator*(const Matrix& matrix, const double* vector)
{
    auto* result = new double[matrix.get_row()];
    for (int i{0}; i < matrix.get_row(); ++i) {
        double value{0};
        for (int j{0}; j < matrix.get_column(); ++j) {
            value += matrix.element(i, j) * vector[j];
        }
        result[i] = value;
    }
    return result;
}

double* operator*(const double* vector, const Matrix& matrix)
{
    auto* result = new double[matrix.get_column()];
    for (int j{0}; j < matrix.get_column(); ++j) {
        double value{0};
        for (int i{0}; i < matrix.get_row(); ++i) {
            value += matrix.element(i, j) * vector[i];
        }
        result[j] = value;
    }
    return result;
}
