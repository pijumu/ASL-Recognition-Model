cmake_minimum_required(VERSION 3.1)
project( ASL-Recognition-model )

# ASL-Recognition-model
find_package( OpenCV REQUIRED )
add_subdirectory( external/yaml-cpp )
include_directories( include ${OpenCV_INCLUDE_DIRS} )
add_executable( ASL-Recognition-model
        source.cpp
    src/Activation.cpp
    src/Derivative.cpp
    src/Matrix.cpp
    src/Network.cpp
    src/read_weights.cpp
    src/write_weights.cpp
)
target_link_libraries( ASL-Recognition-model yaml-cpp ${OpenCV_LIBS} )

# Tests
add_subdirectory( external/doctest )
add_executable(matrix-test src/Matrix.cpp test/test_functions.cpp)
add_executable(derivative-test test/derivative_test.cpp src/Matrix.cpp test/test_functions.cpp src/Derivative.cpp)
add_executable(activation-test test/activation_test.cpp src/Matrix.cpp test/test_functions.cpp src/Activation.cpp)
add_executable(read_weights-test test/read_weights_test.cpp src/Network.cpp test/test_functions.cpp src/Activation.cpp src/Derivative.cpp src/Matrix.cpp src/read_weights.cpp src/write_weights.cpp)

target_link_libraries(matrix-test doctest)
target_link_libraries(derivative-test doctest)
target_link_libraries(activation-test doctest)
target_link_libraries(read_weights-test doctest yaml-cpp::yaml-cpp)


enable_testing()
add_test(NAME matrix-test COMMAND matrix-test —force-colors -d)
add_test(NAME derivative-test COMMAND derivative-test —force-colors -d)
add_test(NAME activation-test COMMAND activation-test —force-colors -d)
add_test(NAME read_weights-test COMMAND read_weights-test —force-colors -d)