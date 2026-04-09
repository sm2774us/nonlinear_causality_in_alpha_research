cmake_minimum_required(VERSION 3.28) # Required for C++ Modules
project(AlphaEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Clang 19 specific flags for std.simd and implicit modules
add_compile_options(
    -fexperimental-library
    -fimplicit-modules
    -fimplicit-module-maps
    -fbuiltin-module-map
    "-I/usr/lib/llvm-19/lib/clang/19/include"
)

# Define the Module Library
add_library(alpha_engine)

# Modern CMake FILE_SET handles the .ixx -> .pcm compilation and propagation
target_sources(alpha_engine
    PUBLIC
        FILE_SET CXX_MODULES FILES
            RiskKernel.ixx
            MacroAlphaEngine.ixx
)

# Shared Library for Python Bindings
add_library(alpha_engine_cpp SHARED bindings.cpp)
target_link_libraries(alpha_engine_cpp PRIVATE alpha_engine)

# Note: Nanobind/GTest dependencies would be passed via Bazel 
# to CMake via CMAKE_PREFIX_PATH in the BUILD file.