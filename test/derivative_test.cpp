#include "include/test_functions.h"
TEST_CASE("Testing derivatives") {
    double* neurons = new double[5]{0.008, -0.003, 0.2, -0.05, 0.00013};
    SUBCASE("Derivative of relu") {
        double* result = der::relu(neurons, 5);
        double* expected = new double[5] {0.1, 0, 0.1, 0, 0.1};
        CHECK(check_equal_vectors(result, expected, 5));
        delete[] expected;
    }
    SUBCASE("Derivative of sigmoid") {
        double* result = der::sigmoid(neurons, 5);
        double* expected = new double[5] {0.007936, -0.003009, 0.16, -0.0525, 0.0001299831};
        CHECK(check_equal_vectors(result, expected, 5));
        delete[] expected;
    }
    SUBCASE("Derivative of softmax") {
        Matrix result = der::softmax(neurons, 5);
        double** elem_exp = new double*[5];
        elem_exp[0] = new double[5]{0.007936, 0.000024, -0.0016, 0.0004, -0.00000104};
        elem_exp[1] = new double[5]{0.000024, -0.003009, 0.0006, -0.00015, 0.00000039};
        elem_exp[2] = new double[5]{-0.0016, 0.0006, 0.16, 0.01, -0.000026};
        elem_exp[3] = new double[5]{0.0004, -0.00015, 0.01, -0.0525, 0.0000065};
        elem_exp[4] = new double[5]{-0.00000104, 0.00000039, -0.000026, 0.0000065, 0.0001299831};
        Matrix expected{5, 5, elem_exp};
        CHECK(result == expected);
    }
    delete[] neurons;
    
}