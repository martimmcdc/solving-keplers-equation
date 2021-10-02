cmake_minimum_required(VERSION 3.4)
project(cpp_implementations)

add_subdirectory(pybind11)
pybind11_add_module(cpp_solvers solvers.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast")