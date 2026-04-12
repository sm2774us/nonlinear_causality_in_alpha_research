cmake_minimum_required(VERSION 3.28)
project(AlphaEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CRITICAL FIX: Set module scanning OFF globally.
# When ON globally, cmake runs clang-scan-deps on ALL .cpp files (nanobind,
# gtest). The -fno-implicit-modules flag then blocks Clang's own builtin
# modules (_Builtin_stdint, _Builtin_stddef) that libc++ headers need -> fatal.
# We enable scanning per-target only on alpha_engine which has .ixx files.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Global ABI flags only — safe for every target including third-party code.
# -fno-implicit-modules / -fno-implicit-module-maps are intentionally absent:
# those flags disable Clang builtin module resolution and break libc++ scanning.
# -march=native must appear here — ALL targets (alpha_engine, alpha_engine_cpp,
# alpha_engine_test_bin) MUST share identical CPU feature flags. Clang embeds
# the target feature set in each .pcm (precompiled module interface). When a
# consumer TU imports a .pcm it verifies the feature sets match exactly; any
# mismatch produces fatal 'AST file was compiled with feature +avx2 but the
# current translation unit is not' errors. Moving -march=native to the global
# list guarantees every TU uses the same CPU baseline.
add_compile_options(
    -stdlib=libc++
    -fexperimental-library
    -Wno-include-angled-in-module-purview
    -march=native
)
add_link_options(
    -stdlib=libc++
    -fexperimental-library
    -lc++abi
    -L/usr/lib/llvm-19/lib
    -Wl,-rpath,/usr/lib/llvm-19/lib
)

if(NOT DEFINED NANOBIND_DIR)
    message(FATAL_ERROR "Pass -DNANOBIND_DIR=<path>")
endif()
if(NOT DEFINED GTEST_SOURCE_DIR)
    message(FATAL_ERROR "Pass -DGTEST_SOURCE_DIR=<path>")
endif()
if(NOT DEFINED TSL_INCLUDE_DIR)
    message(FATAL_ERROR "Pass -DTSL_INCLUDE_DIR=<path>")
endif()

execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "Nanobind    : ${NANOBIND_DIR}")
message(STATUS "GoogleTest  : ${GTEST_SOURCE_DIR}")
message(STATUS "TSL include : ${TSL_INCLUDE_DIR}")
message(STATUS "Python inc  : ${PYTHON_INCLUDE_DIR}")

# GoogleTest — plain C++, no module scanning
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${GTEST_SOURCE_DIR} ${CMAKE_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)

# Nanobind — plain C++ static lib, no module scanning
add_library(nanobind_impl STATIC ${NANOBIND_DIR}/src/nb_combined.cpp)
target_include_directories(nanobind_impl PUBLIC
    ${NANOBIND_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${TSL_INCLUDE_DIR}
)
target_compile_options(nanobind_impl PRIVATE -fPIC -w)
set_target_properties(nanobind_impl PROPERTIES CXX_SCAN_FOR_MODULES OFF)

# Alpha Engine — the ONLY target with C++26 named module units (.ixx)
add_library(alpha_engine STATIC)
target_sources(alpha_engine
    PUBLIC FILE_SET CXX_MODULES FILES
        RiskKernel.ixx
        MacroAlphaEngine.ixx
)
# -march=native removed: it is now in global add_compile_options above.
# Keeping it here (only on alpha_engine) created a CPU feature mismatch: the
# .pcm files for AlphaPod.RiskKernel / AlphaPod.MacroAlphaEngine embedded AVX2
# features, but alpha_engine_cpp and alpha_engine_test_bin lacked them → fatal.
target_compile_options(alpha_engine PRIVATE -O3 -fPIC)
set_target_properties(alpha_engine PROPERTIES CXX_SCAN_FOR_MODULES ON)

# Python bindings — this TU uses 'export module AlphaPod.Bindings' and
# 'import AlphaPod.MacroAlphaEngine/RiskKernel', so CMake MUST scan it for
# module dependencies (CXX_SCAN_FOR_MODULES ON).  Without scanning, Ninja
# never generates the dyndep file and Clang never receives
# -fmodule-file=AlphaPod.MacroAlphaEngine=<path>.pcm → fatal "module not found".
add_library(alpha_engine_cpp SHARED bindings.cpp)
target_include_directories(alpha_engine_cpp PRIVATE
    ${NANOBIND_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${TSL_INCLUDE_DIR}
)
target_link_libraries(alpha_engine_cpp PRIVATE alpha_engine nanobind_impl)
target_compile_options(alpha_engine_cpp PRIVATE -O3 -fPIC)
set_target_properties(alpha_engine_cpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    CXX_SCAN_FOR_MODULES ON
)

# GoogleTest binary
add_executable(alpha_engine_test_bin alpha_engine_test.cpp)
target_link_libraries(alpha_engine_test_bin PRIVATE alpha_engine gtest_main gtest)
# alpha_engine_test.cpp also imports AlphaPod modules — must scan so
# CMake generates dyndep and the compiler finds the .pcm files.
set_target_properties(alpha_engine_test_bin PROPERTIES CXX_SCAN_FOR_MODULES ON)

install(TARGETS alpha_engine_cpp alpha_engine_test_bin DESTINATION .)