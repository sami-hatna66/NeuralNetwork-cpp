add_executable(utils_bench utils_bench.cpp)
target_link_libraries(utils_bench nn-cpp)

add_executable(layer_bench layer_bench.cpp)
target_link_libraries(layer_bench nn-cpp)

add_executable(system_bench system_bench.cpp)
target_link_libraries(system_bench nn-cpp)