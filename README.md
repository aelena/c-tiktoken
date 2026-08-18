# c-tiktoken

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![C Standard](https://img.shields.io/badge/C-C23-orange)
![Integration Tests](https://img.shields.io/badge/integration%20tests-Python%20tiktoken-yellow)

A C implementation of OpenAI's tiktoken tokenizer, built from scratch as an educational tutorial.

## Overview

A working implementation of the tiktoken tokenizer in C, and an eight-chapter tutorial that builds it from base64 decoding up to a complete encoder. The tutorial is the point; the library is what it produces.

## What is tiktoken?

Tiktoken is OpenAI's fast BPE (Byte Pair Encoding) tokenizer used in GPT models. It converts text into sequences of integer tokens that can be processed by language models. This implementation does the same job in C, against the same vocabulary files, and is written to be read rather than to win benchmarks.

## What it does

Encodes and decodes text with OpenAI's `cl100k_base`, or any other `.tiktoken`
vocabulary, in about 1,800 lines of C23 with one dependency.

- **BPE encode and decode.** Byte-level, so any input round-trips exactly,
  including invalid UTF-8.
- **Vocabulary loading** from a file or a memory buffer. Base64 and rank parsing,
  with malformed lines skipped rather than fatal.
- **Robin Hood hash maps** in both directions, bytes to rank and rank to bytes.
  At the 70% load factor the table grows at, that is ~1.37 probes for a hit and
  ~2.37 for a miss.
- **Arena allocation** for token bytes, so a 100K-entry vocabulary is one
  allocation and one `free` rather than 100K of each.
- **PCRE2 pre-tokenization**, with the cl100k, o200k and p50k patterns supplied.
- **Special tokens** either emitted as their own id or treated as ordinary text.
- **Nine test binaries, 94 assertions**, no test framework. Clean under `-Wall
  -Wextra -Wpedantic -Wconversion -Wsign-conversion`.

### What it is not

Worth saying plainly rather than leaving a reader to find out:

- **Not benchmarked against the Rust implementation.** It is written to be read.
  The BPE merge loop is the naive O(n²) one, which is fine for the short chunks
  the regex produces, and Chapter 4 explains what the O(n log n) version would
  cost you in clarity.
- **`TIKTOKEN_SPECIAL_DISALLOW` does not disallow anything yet.** It currently
  behaves as `ALLOW`. The stricter version needs an error channel the API does
  not have.
- **No training.** This encodes with a vocabulary someone else trained.
- **Safe to share, not to build, across threads.** A `TiktokenEncoding` is
  read-only once constructed.
- **`data/` ships empty.** The vocabulary is OpenAI's to distribute, not mine;
  there is a `curl` command below.

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

Two kinds of test: unit tests over synthetic fixtures, and one integration suite that compares this implementation against the official Python `tiktoken` package token for token.

### Test Suites

#### Unit Tests

The unit tests verify each component in isolation using synthetic test data:

- **`test_base64`**. Base64 decoding functionality
  - Tests decoding of various base64 encodings
  - Validates padding handling and error cases
  
- **`test_bytes`**. Byte string operations
  - Tests dynamic byte arrays, slicing, and memory management
  - Validates UTF-8 handling and byte manipulation
  
- **`test_hash`**. Hash map implementation
  - Tests Robin Hood hashing algorithm
  - Validates insertion, lookup, and resizing behavior
  
- **`test_bpe`**. Byte Pair Encoding algorithm
  - Tests BPE merge operations with hand-crafted vocabularies
  - Validates encoding and decoding roundtrips
  
- **`test_regex`**. Regex pre-tokenization
  - Tests PCRE2 integration and pattern matching
  - Validates cl100k_base pattern behavior (contractions, numbers, Unicode)
  
- **`test_vocab`**. Vocabulary loading
  - Tests parsing of .tiktoken vocabulary files
  - Validates base64 decoding and rank parsing
  
- **`test_encoding`**. High-level encoding API
  - Tests complete encode/decode pipeline
  - Validates special token handling and roundtrip encoding

#### Integration Tests

**`test_integration`.** Validates against the official Python tiktoken library.

This is the only test that can tell you the implementation is *correct* rather than merely self-consistent. It:

1. Encode text using the C implementation
2. Call the official Python tiktoken library to get expected results
3. Compare token IDs byte-for-byte to ensure perfect compatibility

**Cases currently compared:**
- `Hello, world!`
- `Hello, world! How are you?` for punctuation
- `Hello 世界 🌍` for multi-byte UTF-8
- `Hello<|endoftext|>world` for special tokens mixed with ordinary text

Four cases, against the real `cl100k_base` vocabulary that GPT-4 uses. Extending
the list is one line each in `tests/test_integration.c`, and worth doing: empty
input, whitespace runs and contractions are all untested against the reference.

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

Without the vocabulary file or the Python package the test exits 77, which CMake is told to read as a skip, so `ctest` reports it as `***Skipped` rather than counting it as a pass. That distinction matters: a green tick from a test that ran no assertions is worse than a red one.

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

Eight chapters, each one component, each building on the last. They are the reason this repository exists.

### Tutorial Chapters

1. **[Chapter 1: Base64 Decoding](tutorial/chapter01_base64.md)**. Learn how to decode base64-encoded vocabulary files
2. **[Chapter 2: Bytestrings](tutorial/chapter02_bytestrings.md)**. Work with byte sequences and UTF-8 handling
3. **[Chapter 3: Hash Map](tutorial/chapter03_hashmap.md)**. Implement a hash map for fast token lookups
4. **[Chapter 4: BPE Algorithm](tutorial/chapter04_bpe.md)**. Understand and implement Byte Pair Encoding
5. **[Chapter 5: Regex](tutorial/chapter05_regex.md)**. Use regex patterns for token splitting
6. **[Chapter 6: Vocabulary](tutorial/chapter06_vocab.md)**. Load and manage token vocabularies
7. **[Chapter 7: API Design](tutorial/chapter07_api.md)**. Compose eight modules into one header worth including
8. **[Chapter 8: Integration](tutorial/chapter08_integration.md)**. Put it all together into a complete tokenizer

**Start with [Chapter 1](tutorial/chapter01_base64.md) to begin the tutorial.**

### As a book

The same eight chapters are also available as a typeset 54-page PDF, *Build a Tokenizer in C*, at **[aelena74.gumroad.com](https://aelena74.gumroad.com)**.

The chapters above are free and stay free. The PDF is the convenience version (front matter, one continuous document, proper typography) and you can build it yourself from [`book/`](book/) if you would rather spend five minutes than the price of a coffee. The instructions there are complete and not deliberately hobbled.

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
        printf("Token %zu: %u\n", i, tokens.items[i]);
    }
    
    // Cleanup
    tokvec_free(&tokens);
    tiktoken_free(enc);
    return 0;
}
```

## API Documentation

The main API is defined in `include/tiktoken/tiktoken.h`. Key functions:

- `tiktoken_from_file()`: Create an encoding from a vocabulary file
- `tiktoken_encode_ordinary()`: Encode text into tokens
- `tiktoken_encode()`: Encode with special token handling
- `tiktoken_decode()`: Decode tokens back to text
- `tiktoken_free()`: Free encoding resources

See the header files in `include/tiktoken/` for detailed documentation.

## License

MIT License. See the LICENSE file for details.

## Contributing

This is an educational project. Feel free to:
- Report bugs or issues
- Suggest improvements to the tutorial
- Submit pull requests with fixes or enhancements

## Acknowledgments

This implementation is based on OpenAI's [tiktoken](https://github.com/openai/tiktoken) Python library and follows the same tokenization algorithm.
