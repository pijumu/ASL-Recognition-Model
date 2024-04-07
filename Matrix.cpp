#include "matrix.h"

//Заполняем матрицу произвольными числам
void Matrix::init(int row, int column) {
    this -> row = row;
    this -> column = column;
    this -> elements = new double* [row];
    for (int i{0}; i < row; ++i) {
        elements[i] = new double[column];
    }
    for (int i{0}; i < row; ++i){
        for (int j{0}; j < column; ++j) {
            elements[i][j] = ((std::rand() % 100)) * 0.007 / (row + column);
        }
    }
};

//Находим сумму двух векторов
void Matrix::sum_vector(double* vector_1, const double* vector_2, int size) {
    for (int i{0}; i < size; ++i) {
        vector_1[i] += vector_2[i];
    }
};


//Находим произведение матрицы на вектор
void Matrix::multiply(Matrix& matrix, const double* vector, int size, double* result) {
    for (int i{0}; i < matrix.row; ++i) {
        double value{0};
        for (int j{0}; j < matrix.column; ++j) {
            value += matrix.elements[i][j] * vector[j];
        }
        result[i] = value;
    }
};

//Получаем количество строк матрицы
int Matrix::get_row() {
    return Matrix::row;
};

//Получаем количество столбцов матрицы
int Matrix::get_column() {
    return Matrix::column;
};

//Получаем элемент матрицы (i, j)
double Matrix::element(int i, int j){
    return Matrix::elements[i][j];
};