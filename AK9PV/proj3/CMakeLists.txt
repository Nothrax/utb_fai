cmake_minimum_required(VERSION 3.21)
project(proj3 CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_executable(proj3 main.cu)

set_target_properties(proj3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(proj3 ${PNG_LIBRARY})

