
#include "Network.h"

void Network::write_weights(const std::string& path) {
    std::ofstream fout("../" + path);
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "network size";
    emitter << YAML::Value << size;

    emitter << YAML::Key << "layers";
    emitter<< YAML::BeginSeq;
    for (int index{0}; index < size; ++index) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "activate function";
        emitter << YAML::Value << layers[index].act_func;
        emitter << YAML::Key << "weights";
        emitter<<YAML::Value << YAML::BeginSeq;
        for (int i{0}; i < layers[index].weights.get_row(); ++i) {
            emitter <<YAML::Value << YAML::Flow;
            emitter <<YAML::Value << YAML::BeginSeq;
            for (int j{0}; j < layers[index].weights.get_column(); ++j) {
                emitter << layers[index].weights.element(i, j);
            }
            emitter << YAML::EndSeq;
        }
        emitter<<YAML::Value << YAML::EndSeq;
        emitter << YAML::Key << "bias";
        emitter << YAML::Flow;
        emitter << YAML::Value << YAML::BeginSeq;
        for (int i{0}; i < layers[index].size; ++i) {
            emitter << layers[index].bias_weights[i];
        }
        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;
    }
    emitter<< YAML::EndSeq;
    emitter<< YAML::EndSeq;
    fout << emitter.c_str();
    fout.close();

    
};