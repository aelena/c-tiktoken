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

add_executable(test_integration tests/test_integration.c)
target_link_libraries(test_integration PRIVATE tiktoken)
add_test(NAME integration COMMAND test_integration)

# The integration suite needs cl100k_base.tiktoken and the Python tiktoken
# package, neither of which ships here. Without this property it exited 0 when
# it could not run, and ctest reported the one test that checks us against the
# reference implementation as passing. 77 is the Automake convention for a skip.
set_tests_properties(integration PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_roundtrip tests/test_roundtrip.c)
target_link_libraries(test_roundtrip PRIVATE tiktoken)
add_test(NAME roundtrip COMMAND test_roundtrip)

# Volume comparison against the Python package. Same inputs as the integration
# suite needs, so the same skip convention applies.
add_executable(test_reference_batch tests/test_reference_batch.c)
target_link_libraries(test_reference_batch PRIVATE tiktoken)
add_test(NAME reference_batch COMMAND test_reference_batch)
set_tests_properties(reference_batch PROPERTIES SKIP_RETURN_CODE 77)

# Fuzz targets
#
# Off by default and clang-only: libFuzzer ships with clang, not gcc. The targets
# link the library and let libFuzzer drive the two functions that parse bytes this
# project did not write.
#
#     cmake -S . -B fz -DCMAKE_C_COMPILER=clang-19 -DTIKTOKEN_FUZZ=ON
#     cmake --build fz
#     ./fz/fuzz_vocab -max_total_time=60 fuzz/corpus/vocab
option(TIKTOKEN_FUZZ "Build the libFuzzer targets (requires clang)" OFF)

if(TIKTOKEN_FUZZ)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "TIKTOKEN_FUZZ needs clang; libFuzzer does not ship with gcc.")
    endif()

    # The library itself has to carry the coverage instrumentation, not just the
    # harness. Without this libFuzzer sees coverage only inside the fuzz target,
    # gets no signal from the code under test, and degenerates into random input
    # generation: thirty million executions at a coverage of 4 features, which
    # looks like throughput and is worth nothing.
    target_compile_options(tiktoken PRIVATE
        -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g)
    target_link_options(tiktoken PUBLIC -fsanitize=address,undefined)

    foreach(target fuzz_base64 fuzz_vocab)
        add_executable(${target} fuzz/${target}.c)
        target_link_libraries(${target} PRIVATE tiktoken)
        target_compile_options(${target} PRIVATE
            -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -g)
        target_link_options(${target} PRIVATE
            -fsanitize=fuzzer,address,undefined)
    endforeach()
endif()

# Examples
add_executable(count_tokens examples/count_tokens.c)
target_link_libraries(count_tokens PRIVATE tiktoken)

add_executable(encode_decode examples/encode_decode.c)
target_link_libraries(encode_decode PRIVATE tiktoken)
