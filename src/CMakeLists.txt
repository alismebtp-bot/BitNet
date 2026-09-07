set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET
    ggml-bitnet-mad.cpp
    ggml-bitnet-lut.cpp
)

# Keep include path rooted at the BitNet project so both mad and lut kernels resolve.
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
