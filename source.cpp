#include "Network.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <cstdlib>
#include <ctime>

namespace fs = std::filesystem;

std::unordered_map<int, std::string> ind_class = {
    {0, "A"},
    {1, "B"},
    {2, "C"},
    {3, "D"},
    {4, "E"},
    {5, "F"},
    {6, "G"},
    {7, "H"},
    {8, "I"},
    {9, "J"},
    {10, "K"},
    {11, "L"},
    {12, "M"},
    {13, "N"},
    {14, "O"},
    {15, "P"},
    {16, "Q"},
    {17, "R"},
    {18, "S"},
    {19, "T"},
    {20, "U"},
    {21, "V"},
    {22, "W"},
    {23, "X"},
    {24, "Y"},
    {25, "Z"},
    {26, "del"},
    {27, "space"},
    {28, "nothing"}
};

std::unordered_map<std::string, int> class_ind = {
    {"A", 0},
    {"B", 1},
    {"C", 2},
    {"D", 3},
    {"E", 4},
    {"F", 5},
    {"G", 6},
    {"H", 7},
    {"I", 8},
    {"J", 9},
    {"K", 10},
    {"L", 11},
    {"M", 12},
    {"N", 13},
    {"O", 14},
    {"P", 15},
    {"Q", 16},
    {"R", 17},
    {"S", 18},
    {"T", 19},
    {"U", 20},
    {"V", 21},
    {"W", 22},
    {"X", 23},
    {"Y", 24},
    {"Z", 25},
    {"del", 26},
    {"space", 27},
    {"nothing", 28}
};

int main() {
    std::srand(std::time(nullptr));
    YAML::Node settings = YAML::LoadFile("../all_yaml_configs/settings_train.yaml");
    auto network_cfg_path = settings["network_cfg_path"].as<std::string>();
    const auto train_or_predict = settings["train_or_predict"].as<std::string>();
    const auto data_folder = ".." + settings["data_folder"].as<std::string>();
    const int n = data_folder.size();
    Network asl(network_cfg_path, train_or_predict);
    if (train_or_predict == "train") {
        for (const auto &entry: fs::directory_iterator(data_folder)) {
            for (const auto &img: fs::directory_iterator(entry)) {
                constexpr int height = 40;
                constexpr int width = 40;
                cv::Mat image = cv::imread(img.path());
                resize(image, image, cv::Size(width, height), cv::INTER_LINEAR);
                cv::imwrite(img.path(), image);
            }
        }
        const auto output_file = settings["output_file"].as<std::string>();
        const int epochs = settings["epochs"].as<int>();
        const int batch_size = settings["batch_size"].as<int>();
        auto *input = new double[1600];
        auto *ans = new double[29];
        int our_batch = 0;
        double *lrs;
        lrs = new double[20]{
            0.001, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.001,
            0.0007, 0.0007, 0.0003, 0.0003, 0.0003,
            0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
        };
        asl.dropout_mask();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "starting epoch: " << epoch << '\n';
            double lr = lrs[epoch];
            for (int k = 1; k < 1501; ++k) {
                for (const auto &entry: fs::directory_iterator(data_folder)) {
                    std::string p = entry.path();
                    std::string c = p.substr(n + 1, p.size());
                    for (int m = 0; m < 29; ++m) {
                        if (m == class_ind[c]) {
                            ans[m] = 1.0;
                        } else {
                            ans[m] = 0.0;
                        }
                    }
                    p += "/" + p.substr(n + 1, p.size()) + std::to_string(k) + ".jpg";
                    cv::Mat image = cv::imread(p);
                    int input_index = 0;
                    for (int i = 0; i < 40; ++i) {
                        for (int j = 0; j < 40; ++j) {
                            const double intensity = image.at<uchar>(i, j);
                            input[input_index] = intensity / 255;
                            ++input_index;
                        }
                    }
                    ++our_batch;
                    asl.set_input(input);
                    asl.forward_feed();
                    asl.back_propagation(ans);
                    if (our_batch == batch_size) {
                        our_batch = 0;
                        asl.update_weights(lr);
                        asl.dropout_mask();
                    }
                }
            }
        }
        asl.write_weights(output_file);
        delete [] lrs;
        delete [] input;
        delete [] ans;
    } else {
        for (const auto &img: fs::directory_iterator(data_folder)) {
            constexpr int height = 40;
            constexpr int width = 40;
            cv::Mat image = cv::imread(img.path());
            resize(image, image, cv::Size(width, height), cv::INTER_LINEAR);
            cv::imwrite(img.path(), image);
        }
        auto *input = new double[1600];
        for (const auto &img: fs::directory_iterator(data_folder)) {
            cv::Mat image = cv::imread(img.path());
            int input_index = 0;
            std::string p = img.path();
            p = p.substr(n + 1, p.size());
            for (int i = 0; i < 40; ++i) {
                for (int j = 0; j < 40; ++j) {
                    const double intensity = image.at<uchar>(i, j);
                    input[input_index] = intensity / 255;
                    ++input_index;
                }
            }
            asl.set_input(input);
            asl.forward_feed();
            std::cout << "predicted: " << ind_class[asl.predict()] << '\n';
            std::cout << "expected: " << p << '\n';
            std::cout << '\n';
        }
        delete [] input;
    }
}
