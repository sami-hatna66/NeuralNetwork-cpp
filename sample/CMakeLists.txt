find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mnistsample MnistSample.cpp)
target_link_libraries(mnistsample nn-cpp ${OpenCV_LIBS})
