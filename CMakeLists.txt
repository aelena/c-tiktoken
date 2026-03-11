cmake_minimum_required(VERSION 3.25)
project(c-tiktoken VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Strict warnings
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wnull-dereference
)

# Library
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c
    src/hash.c
    src/bpe.c
    src/regex.c
    src/vocab.c
    src/encoding.c
)
target_include_directories(tiktoken PUBLIC include)

# PCRE2 dependency
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PCRE2 IMPORTED_TARGET libpcre2-8)
endif()
if(PCRE2_FOUND)
    target_link_libraries(tiktoken PUBLIC PkgConfig::PCRE2)
else()
    find_library(PCRE2_LIB pcre2-8)
    find_path(PCRE2_INC pcre2.h)
    if(PCRE2_LIB AND PCRE2_INC)
        target_link_libraries(tiktoken PUBLIC ${PCRE2_LIB})
        target_include_directories(tiktoken PUBLIC ${PCRE2_INC})
    else()
        message(FATAL_ERROR "PCRE2 not found. Install libpcre2-dev (apt), pcre2 (brew), or pcre2 (vcpkg).")
    endif()
endif()

# Tests
enable_testing()

add_executable(test_base64 tests/test_base64.c)
target_link_libraries(test_base64 PRIVATE tiktoken)
add_test(NAME base64 COMMAND test_base64)

add_executable(test_bytes tests/test_bytes.c)
target_link_libraries(test_bytes PRIVATE tiktoken)
add_test(NAME bytes COMMAND test_bytes)

add_executable(test_hash tests/test_hash.c)
target_link_libraries(test_hash PRIVATE tiktoken)
add_test(NAME hash COMMAND test_hash)

add_executable(test_bpe tests/test_bpe.c)
target_link_libraries(test_bpe PRIVATE tiktoken)
add_test(NAME bpe COMMAND test_bpe)

add_executable(test_regex tests/test_regex.c)
target_link_libraries(test_regex PRIVATE tiktoken)
add_test(NAME regex COMMAND test_regex)

add_executable(test_vocab tests/test_vocab.c)
target_link_libraries(test_vocab PRIVATE tiktoken)
add_test(NAME vocab COMMAND test_vocab)

add_executable(test_encoding tests/test_encoding.c)
target_link_libraries(test_encoding PRIVATE tiktoken)
add_test(NAME encoding COMMAND test_encoding)

# Examples
add_executable(count_tokens examples/count_tokens.c)
target_link_libraries(count_tokens PRIVATE tiktoken)

add_executable(encode_decode examples/encode_decode.c)
target_link_libraries(encode_decode PRIVATE tiktoken)
