# c-tiktoken

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![C Standard](https://img.shields.io/badge/C-C23-orange)
![Integration Tests](https://img.shields.io/badge/integration%20tests-Python%20tiktoken-yellow)

A C implementation of OpenAI's tiktoken tokenizer, built from scratch as an educational tutorial.

## Overview

This repository contains a complete, production-ready implementation of the tiktoken tokenizer in C. The project is structured as a step-by-step tutorial that guides you through building a tokenizer from the ground up, covering everything from base64 decoding to the final integration.

## What is tiktoken?

Tiktoken is OpenAI's fast BPE (Byte Pair Encoding) tokenizer used in GPT models. It converts text into sequences of integer tokens that can be processed by language models. This implementation provides the same functionality in pure C, making it suitable for embedded systems, high-performance applications, or as a learning resource.

## Features

- **Complete BPE tokenizer implementation** — Full support for OpenAI's tokenization algorithm
- **Base64 vocabulary decoding** — Efficient parsing of tiktoken vocabulary files
- **Regex-based token splitting** — PCRE2-powered pattern matching for token boundaries
- **Memory-efficient arena allocator** — Custom memory management for performance
- **Hash map implementation** — Fast token lookup and merging
- **C23 standard** — Modern C features for cleaner, safer code
- **Comprehensive test suite** — Unit tests for all components
- **Example programs** — Ready-to-use tokenization examples

## Project Structure

```
c-tiktoken/
├── include/tiktoken/    # Public API headers
├── src/                 # Implementation source files
├── tests/               # Unit tests
├── examples/            # Example programs
├── tutorial/            # Step-by-step tutorial chapters
└── data/                # Vocabulary and data files
```

## Building

### Prerequisites

- CMake 3.25 or later
- C23-compatible compiler (GCC 13+, Clang 16+)
- PCRE2 library (libpcre2-dev on Debian/Ubuntu, pcre2 on macOS/Homebrew)

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

## Tests

This project includes a comprehensive test suite covering all components of the tokenizer implementation. The tests are organized into two categories: **unit tests** for individual components and **integration tests** that validate against the official Python tiktoken library.

### Test Suites

#### Unit Tests

The unit tests verify each component in isolation using synthetic test data:

- **`test_base64`** — Base64 decoding functionality
  - Tests decoding of various base64 encodings
  - Validates padding handling and error cases
  
- **`test_bytes`** — Byte string operations
  - Tests dynamic byte arrays, slicing, and memory management
  - Validates UTF-8 handling and byte manipulation
  
- **`test_hash`** — Hash map implementation
  - Tests Robin Hood hashing algorithm
  - Validates insertion, lookup, and resizing behavior
  
- **`test_bpe`** — Byte Pair Encoding algorithm
  - Tests BPE merge operations with hand-crafted vocabularies
  - Validates encoding and decoding roundtrips
  
- **`test_regex`** — Regex pre-tokenization
  - Tests PCRE2 integration and pattern matching
  - Validates cl100k_base pattern behavior (contractions, numbers, Unicode)
  
- **`test_vocab`** — Vocabulary loading
  - Tests parsing of .tiktoken vocabulary files
  - Validates base64 decoding and rank parsing
  
- **`test_encoding`** — High-level encoding API
  - Tests complete encode/decode pipeline
  - Validates special token handling and roundtrip encoding

#### Integration Tests

**`test_integration`** — Validates against official Python tiktoken library

These tests are the **gold standard** for correctness. They:

1. Encode text using the C implementation
2. Call the official Python tiktoken library to get expected results
3. Compare token IDs byte-for-byte to ensure perfect compatibility

**Test Coverage:**
- Simple text encoding
- Empty strings and edge cases
- Numbers, punctuation, and contractions
- Unicode characters (e.g., "Hello 世界 🌍")
- Newlines and whitespace handling
- Special tokens (`<|endoftext|>`, etc.)
- Mixed special and ordinary text

The integration tests use the real `cl100k_base` vocabulary file (used by GPT-4) and ensure that the C implementation produces **identical** tokenization results to OpenAI's official library.

### Running Tests

#### All Unit Tests

```bash
cd build
ctest
```

This runs all unit tests via CMake's test framework.

#### Integration Tests

The integration tests require Python and the official tiktoken library:

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the vocabulary file (if not already present):
   ```bash
   curl -o data/cl100k_base.tiktoken \
     https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
   ```

3. Run the integration test:
   ```bash
   cd build
   ./test_integration
   ```

The integration test will automatically skip if the Python reference script or vocabulary file is not available, making it safe to run in CI/CD environments where Python might not be installed.

### Test Philosophy

- **Unit tests** use synthetic data to verify algorithm correctness
- **Integration tests** use real vocabulary files and compare against the official implementation
- All tests are self-contained and don't require external test frameworks
- Tests provide clear pass/fail output with detailed error messages

### Running Examples

```bash
cd build
./count_tokens "Hello, world!"
./encode_decode
```

## Tutorial

This repository includes a comprehensive tutorial that walks you through building the tokenizer step by step. Each chapter focuses on a specific component and builds upon the previous ones.

### Tutorial Chapters

1. **[Chapter 1: Base64 Decoding](tutorial/chapter01_base64.md)** — Learn how to decode base64-encoded vocabulary files
2. **[Chapter 2: Bytestrings](tutorial/chapter02_bytestrings.md)** — Work with byte sequences and UTF-8 handling
3. **[Chapter 3: Hash Map](tutorial/chapter03_hashmap.md)** — Implement a hash map for fast token lookups
4. **[Chapter 4: BPE Algorithm](tutorial/chapter04_bpe.md)** — Understand and implement Byte Pair Encoding
5. **[Chapter 5: Regex](tutorial/chapter05_regex.md)** — Use regex patterns for token splitting
6. **[Chapter 6: Vocabulary](tutorial/chapter06_vocab.md)** — Load and manage token vocabularies
7. **[Chapter 7: API Design](tutorial/chapter07_api.md)** — Design a clean, user-friendly API
8. **[Chapter 8: Integration](tutorial/chapter08_integration.md)** — Put it all together into a complete tokenizer

**Start with [Chapter 1](tutorial/chapter01_base64.md) to begin the tutorial.**

## Usage

### Basic Example

```c
#include <tiktoken/tiktoken.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    // Load encoding from file
    const SpecialToken *special;
    size_t n_special = tiktoken_cl100k_special(&special);
    
    TiktokenEncoding *enc = tiktoken_from_file(
        "cl100k_base.tiktoken",
        tiktoken_pattern_cl100k(),
        special,
        n_special
    );
    
    if (enc == nullptr) {
        fprintf(stderr, "Failed to load encoding\n");
        return 1;
    }
    
    // Encode text
    const char *text = "Hello, world!";
    TokenVec tokens = tiktoken_encode_ordinary(enc, text, strlen(text));
    
    printf("Token count: %zu\n", tokens.len);
    for (size_t i = 0; i < tokens.len; i++) {
        printf("Token %zu: %u\n", i, tokens.data[i]);
    }
    
    // Cleanup
    tokvec_free(&tokens);
    tiktoken_free(enc);
    return 0;
}
```

## API Documentation

The main API is defined in `include/tiktoken/tiktoken.h`. Key functions:

- `tiktoken_from_file()` — Create an encoding from a vocabulary file
- `tiktoken_encode_ordinary()` — Encode text into tokens
- `tiktoken_encode()` — Encode with special token handling
- `tiktoken_decode()` — Decode tokens back to text
- `tiktoken_free()` — Free encoding resources

See the header files in `include/tiktoken/` for detailed documentation.

## License

MIT License — see LICENSE file for details.

## Contributing

This is an educational project. Feel free to:
- Report bugs or issues
- Suggest improvements to the tutorial
- Submit pull requests with fixes or enhancements

## Acknowledgments

This implementation is based on OpenAI's [tiktoken](https://github.com/openai/tiktoken) Python library and follows the same tokenization algorithm.
