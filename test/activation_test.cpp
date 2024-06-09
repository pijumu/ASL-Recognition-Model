#include "include/test_functions.h"

TEST_CASE("Testing activate functions") {
    double* neurons = new double[5]{0.008, -0.003, 0.2, -0.05, 0.00013};
    SUBCASE("Activate function of softmax") {
        double* result = act::softmax(neurons, 5);
        double* expected = new double[5] {0.194684, 0.192554, 0.235892, 0.183713, 0.193157};
        CHECK(check_equal_vectors(result, expected, 5));
        delete[] expected;
    }
    SUBCASE("Activate function of sigmoid") {
        double* result = act::sigmoid(neurons, 5);
        double* expected = new double[5] {0.502, 0.49925, 0.549834, 0.487503, 0.500032};
        CHECK(check_equal_vectors(result, expected, 5));
        delete[] expected;
    }
    SUBCASE("Activate function of relu") {
        double* result = act::relu(neurons, 5);
        double* expected = new double[5] {0.0008, 0, 0.02, 0, 0.000013};
        CHECK(check_equal_vectors(result, expected, 5));
        delete[] expected;
    }

}