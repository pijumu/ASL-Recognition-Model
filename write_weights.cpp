#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include "Network.h"

/*void NetWork::write_weights(const std::string& path) {
    YAML::Node node = YAML::LoadFile("config.yaml");
    node["afaf"] = "afaf";
    std::string result_path = "../" + path;
    std::ofstream out{result_path};
    out << "network_size: " <<size << "\n";
    int const pixels{16}; //Количество пикселей в картинке
    for (int index{0}; index < size; ++index) { //Проходим все слои (кроме входного слоя)
        out << "- layer" << index << ":\n";
        out << "    size: " <<layers[index]->size << "/n";
        out << "    weights_matrix:\n";
        for (int i{0}; i < layers[index]->weights.get_row(); ++i) {
            out << "row_" << i << ":\n";
            for (int j{0}; j < layers[index]->weights.get_column(); ++j) {
                out << "";
            }
        }

    }
    out.close();
    node["afaf"] = "afaf";
    std::ofstream fout("config.yaml");
    fout << node;*/

void Network::write_weights(const std::string& path) {
    std::ofstream fout("../" + path);
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "network size";
    emitter << YAML::Value << size;

    emitter << YAML::Key << "loss function";
    emitter << YAML::Value << loss_function;

    emitter << YAML::Key << "layers";
    emitter<< YAML::BeginSeq;
    for (int index{0}; index < size; ++index) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "activation function";
        emitter << YAML::Value << layers[index].activation_function;
        emitter << YAML::Key << "weights";
        emitter<<YAML::Value << YAML::BeginSeq;
        for (int i{0}; i < layers[index].size; ++i) {
            emitter <<YAML::Value << YAML::Flow;
            emitter <<YAML::Value << YAML::BeginSeq << 1 << 2 << 3 << 4 << 5 << YAML::EndSeq;
            for (int j{0}; j < )
        }
    }


    
};
int main()
{
    YAML::Emitter emitter;
    emitter << YAML::BeginMap; //
    emitter << YAML::Key << "network size";//
    emitter << YAML::Value << 2;//

    emitter << YAML::Key << "loss function";//
    emitter << YAML::Value << "crossentropy";//

    emitter << YAML::Key << "layers";//
    emitter<< YAML::BeginSeq;//
    for (int i{0}; i < 3; ++i) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "activation function";
        emitter << YAML::Value << "relu";
        emitter << YAML::Key << "weights";
        emitter<<YAML::Value << YAML::BeginSeq;
        for (int j{0}; j < 4; ++j) {
            emitter <<YAML::Value << YAML::Flow;
            emitter <<YAML::Value << YAML::BeginSeq << 1 << 2 << 3 << 4 << 5 << YAML::EndSeq;
        }
        emitter<<YAML::Value << YAML::EndSeq;
        emitter << YAML::Key << "bias";
        emitter << YAML::Flow;
        emitter << YAML::Value << YAML::BeginSeq << 1 << 2 << 3 << 4 << 5 << YAML::EndSeq;
        emitter << YAML::EndMap;

    }
    emitter<< YAML::EndSeq;
    emitter<< YAML::EndSeq;
    std::ofstream fout("../example.yaml");
    fout << emitter.c_str();
    fout.close();
    /*
    YAML::Node config;
    config["key"] = "value";
    fout << config;*/
}