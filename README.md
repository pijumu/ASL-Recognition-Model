# ASL-Recognition-Model

## 📖 About:
Tools for network monitoring
The American Sign Language (ASL) recognition model is a machine learning model trained to accurately interpret and understand hand gestures and movements used in ASL.

## 🔎 Prerequisites:

To compile and run the project, you will need the following C++ libraries:

- [yaml-cpp](https://github.com/jbeder/yaml-cpp) - a library for working with YAML files.
- [doctest](https://github.com/onqtam/doctest) - a C++ testing framework.
- [OpenCV](https://opencv.org/) - a library for computer vision and image processing.

Data for training and testing:
- [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) - Image data set for alphabets in the American Sign Language

## 🔨 Building the Project:

1. Clone this repository:

```
git clone git@github.com:pijumu/ASL-Recognition-Model.git
```

2. Go to project directory:

```
cd ASL-Recognition-Model
```

3. Move yaml-cpp and doctest into external directory:

```
📂ASL-Recognition-model 🔻
    📁all_yaml_configs 
    📁docs
    📂external 🔻
        📁doctest 
        📁yaml-cpp
    ...
```

4. Move installed data to current directory. So it will look like:
```
📂ASL-Recognition-model 🔻
    📁all_yaml_configs 
    📁asl_alphabet_train
    📁asl_alphabet_test
    📁docs
    📁external
    ...
```
5. Make build directory:

```
mkdir build && cd build
```

6. CMake build process:

```
cmake .. && cmake --build .
```

## 🎓 Usage for training:

1. Training config ***all_yaml_configs/settings_train.yaml***:

```yaml
network_cfg_path: all_yaml_configs/network_config_train.yaml
train_or_predict: train
data_folder: /asl_alphabet_train
epochs: 10
batch_size: 17
output_file: all_yaml_configs/save_weights.yaml
```

2. Training config ***all_yaml_configs/settings_train.yaml***:
- Choose dropout probability.
- Choose activation functions: ***relu***, ***sigmoid***, ***softmax***.
- Choose layer sizes.
```yaml
network_size: 2
size_of_initial_neurons: 1600
dropout_probability: 0.5
layers:
- activate_function: relu
  layer_size: 256
- activate_function: softmax
  layer_size: 29
```

3. Change ***source.cpp***:

```c++
YAML::Node settings = YAML::LoadFile("../all_yaml_configs/settings_train.yaml");
...
```

4. Running:

```
cd build
./ASL-Recognition-model
```

## 🎯 Usage for predicting:

1. Predicting config ***all_yaml_configs/settings_predict.yaml***:

```yaml
network_cfg_path: all_yaml_configs/save_weights.yaml
train_or_predict: predict
data_folder: /asl_alphabet_test
```

2. Change ***source.cpp***:

```c++
YAML::Node settings = YAML::LoadFile("../all_yaml_configs/settings_predict.yaml");
...
```

3. Running:

```
cd build
./ASL-Recognition-model
```