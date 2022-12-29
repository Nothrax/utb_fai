CMAKE_MINIMUM_REQUIRED(VERSION 3.21)
PROJECT(simulator)

SET(CMAKE_CXX_STANDARD 17)

FILE(GLOB_RECURSE source_files "source/*")
ADD_LIBRARY(simulator_lib STATIC ${source_files})
TARGET_INCLUDE_DIRECTORIES(simulator_lib PUBLIC "include/")

ADD_EXECUTABLE(simulator main.cpp)

TARGET_LINK_LIBRARIES(simulator simulator_lib)
