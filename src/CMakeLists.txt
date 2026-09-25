cmake_minimum_required(VERSION 3.22)
project(lexa_jni C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED LLAMA_DIR)
    message(FATAL_ERROR "LLAMA_DIR not set")
endif()

set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE       OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

add_library(lexa_jni SHARED
    jni_bridge.cpp
    tools_bridge.cpp
    third_party/miniz/miniz.c
)

set_target_properties(lexa_jni PROPERTIES
    OUTPUT_NAME llama
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(lexa_jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
    third_party/miniz
)

target_link_libraries(lexa_jni PRIVATE
    llama
    ggml
    android
    log
)
