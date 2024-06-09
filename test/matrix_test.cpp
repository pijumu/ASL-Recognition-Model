#include "include/test_functions.h"

TEST_CASE("Operations with matrix and vectors") {
    double** elem_1 = new double*[4];
    elem_1[0] = new double[3]{0.1, 0.2, 0.05};
    elem_1[1] = new double[3]{0.03, 0.08, 0.07};
    elem_1[2] = new double[3]{1, 0.2, 5};
    elem_1[3] = new double[3]{0.2, 0.2, 0.05};
    Matrix matrix_1{4, 3, elem_1};

    double** elem_2 = new double*[3];
    elem_2[0] = new double[3]{0.1, 0.02, 0.1};
    elem_2[1] = new double[3]{0.03, 1, 0.09};
    elem_2[2] = new double[3]{0.1, 2, 5};
    Matrix matrix_2{3, 3, elem_2};

    double* vector_1 = new double[4]{0.008, 0.2, 0.4, 0.12};
    double* vector_2 = new double[3]{0.02, 0.04, 0.5};
    double* vector_3 = new double[4]{1.01, 2.107, 0.002, 0.3};

    SUBCASE("Matrix multiply") {
        double** elem_exp = new double*[4];
        elem_exp[0] = new double[3]{0.021, 0.302, 0.278};
        elem_exp[1] = new double[3]{0.0124, 0.2206, 0.3602};
        elem_exp[2] = new double[3]{0.606, 10.22, 25.118};
        elem_exp[3] = new double[3]{0.031, 0.304, 0.288};
        Matrix expected{4, 3, elem_exp};
        CHECK(matrix_1*matrix_2 == expected);
    }

    SUBCASE("Matrix and vector multiply") {
        double* expected = new double[4]{0.035, 0.0388, 2.528, 0.037};
        CHECK(check_equal_vectors(matrix_1*vector_2, expected, 4));
    }

    SUBCASE("Vector and matrix multiply") {
        double* expected = new double[3]{0.4308, 0.1216, 2.0204};
        CHECK(check_equal_vectors(vector_1*matrix_1, expected, 3));
    }

    SUBCASE("Sum of vectors") {
        double* expected = new double[4]{1.018, 2.307, 0.402, 0.42};
        double* result = Matrix::sum_vector(vector_1, vector_3, 4);
        CHECK(check_equal_vectors(result, expected, 4));
    }

    SUBCASE("Multiplying elements one by one") {
        double* expected = new double[4]{0.00808, 0.4214, 0.008, 0.036};
        double* result = Matrix::multy_elements(vector_1, vector_3, 4);
        CHECK(check_equal_vectors(result, expected, 4));
    }
    delete[] vector_1;
    delete[] vector_2;
    delete[] vector_3;
}

TEST_CASE("Testing constructors of matrix") {
    SUBCASE("Matrix constructor without elements") {
        Matrix matrix(2, 4);
        CHECK(matrix.get_row() == 2);
        CHECK(matrix.get_column() == 4);
    }

    SUBCASE("Matrix constructor with elements") {
        double** elem = new double*[3];
        elem[0] = new double[3]{0.01, 0.3, 0.55};
        elem[1] = new double[3]{0.03, 0.08, 0.07};
        elem[2] = new double[3]{1.02, 0.2, 0.05};
        Matrix matrix{3, 3, elem};
        CHECK(matrix.get_row() == 3);
        CHECK(matrix.get_column() == 3);
        CHECK(matrix.element(0, 0) - 0.01 < 0.0000000001);
        CHECK(matrix.element(0, 2) - 0.55 < 0.0000000001);
        CHECK(matrix.element(2, 1) - 0.2 < 0.0000000001);
        CHECK(matrix.element(2, 2) - 0.05 < 0.0000000001);

    }

}




/*
    for (int i{0}; i < 4; ++i) {
            for (int j{0}; j < 3; ++j) {
                std::cout << a.element(i, j) << " ";
            }
            std::cout << std::endl;
        }*/