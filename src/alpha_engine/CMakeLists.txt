cmake_minimum_required(VERSION 3.28)
project(AlphaEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. ISOLATED FLAGS: Remove -fimplicit-modules to stop the 'pthread.h' warnings.
# Bazel already provides the necessary module maps for libc++.
add_compile_options(-fexperimental-library -stdlib=libc++)

# 2. Engine Library
add_library(alpha_engine STATIC)
target_sources(alpha_engine
    PUBLIC
        FILE_SET CXX_MODULES FILES
            RiskKernel.ixx
            MacroAlphaEngine.ixx
)

# 3. Shared Library (Python Extension)
add_library(alpha_engine_cpp SHARED bindings.cpp)
target_link_libraries(alpha_engine_cpp PRIVATE alpha_engine)

# 4. THE PLUG: Nuclear option for 'multiple definition' linker errors.
# Forces ld.gold to accept the first symbol found for nanobind objects.
if(NOT WIN32)
    target_link_options(alpha_engine_cpp PRIVATE "-Wl,--allow-multiple-definition")
endif()

# 5. Test Binary
add_executable(alpha_engine_test alpha_engine_test.cpp)
target_link_libraries(alpha_engine_test PRIVATE alpha_engine)
if(NOT WIN32)
    target_link_options(alpha_engine_test PRIVATE "-Wl,--allow-multiple-definition")
endif()