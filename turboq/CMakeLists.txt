cmake_minimum_required(VERSION 3.15)
project(turboq_ext LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(turboq_ext
    bindings.cpp
    turboq.c
    codebook.c
    hd3.c
)

target_include_directories(turboq_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(turboq_ext PRIVATE -O3)
