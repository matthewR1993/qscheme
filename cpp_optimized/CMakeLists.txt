cmake_minimum_required(VERSION 3.12)
project(cpp_optimized)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Ofast -frename-registers -funroll-loops -Wall -Wextra -ffast-math -fno-signed-zeros -ffinite-math-only -fno-signed-zeros -fno-trapping-math -fwhole-program -flto -fuse-linker-plugin -fcx-limited-range")

add_executable(cpp_optimized main.cpp)