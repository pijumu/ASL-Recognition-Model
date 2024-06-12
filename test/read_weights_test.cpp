#include "include/test_functions.h"
#include "Network.h"
TEST_CASE("Testing network train constructor") {
    Network network{"test/config_train_test.yaml", "train"};
    CHECK(network.loss_func == CrossEntropy);
    CHECK(network.size == 2);
    CHECK(network.dropout_prob == 0.5);
    CHECK(network.layers.size() == 2);
    CHECK(network.layers[0].act_func == "relu");
    CHECK(network.layers[1].size == 29);
}

TEST_CASE("Testing network predict constructor") {
    Network network{"test/config_predict_test.yaml", "predict"};
    CHECK(network.loss_func == MSE);
    CHECK(network.size == 2);
    double** elements = new double*[3];
    elements[0] = new double[2]{0.047833333333333339, 0.078166666666666676};
    elements[1] = new double[2]{0, 0.080500000000000002};
    elements[2] = new double[2]{0.091000000000000011, 0.067666666666666667};

    Matrix expected{3, 2, elements};
    CHECK(network.layers[0].weights == expected);
    CHECK(network.layers[1].act_func == "sigmoid");
}