# Chapter 8 — Putting It All Together

## What We Built

Over seven chapters, we constructed a complete BPE tokenizer in modern
C23 — compatible with OpenAI's tiktoken format:

| Chapter | Module | Lines of C | What It Does |
|---------|--------|-----------|--------------|
| 1 | `base64` | ~120 | Decode base64 token data from vocabulary files |
| 2 | `bytes` | ~200 | Length-prefixed byte buffers with ownership tracking |
| 3 | `arena` + `hash` | ~350 | Bulk memory allocation + Robin Hood hash maps |
| 4 | `bpe` | ~150 | The core BPE merge algorithm |
| 5 | `regex` | ~130 | PCRE2 wrapper for Unicode-aware text splitting |
| 6 | `vocab` | ~150 | Load .tiktoken vocabulary files |
| 7 | `encoding` | ~170 | High-level encode/decode API with special tokens |
| 8 | `tiktoken.h` | ~50 | Convenience header + preset constructors |
| **Total** | | **~1,320** | A complete tiktoken-compatible tokenizer |

That's roughly 1,300 lines of C for a fully functional tokenizer. The
Python tiktoken library is ~800 lines of Python + ~2,000 lines of Rust
(for the core). Our C version is competitive in size and does everything
from scratch (except regex).

## Using the Library

### Basic Usage

```c
#include <tiktoken/tiktoken.h>

int main(void) {
    // Load the cl100k_base encoding (GPT-4).
    const SpecialToken *special;
    size_t n_special = tiktoken_cl100k_special(&special);

    TiktokenEncoding *enc = tiktoken_from_file(
        "cl100k_base.tiktoken",
        tiktoken_pattern_cl100k(),
        special, n_special
    );

    // Encode text.
    const char *text = "Hello, world!";
    TokenVec tokens = tiktoken_encode_ordinary(enc, text, strlen(text));

    printf("Token count: %zu\n", tokens.len);
    for (size_t i = 0; i < tokens.len; i++) {
        printf("  %u\n", tokens.items[i]);
    }

    // Decode back.
    Bytes decoded = tiktoken_decode(enc, tokens.items, tokens.len);
    printf("Decoded: %.*s\n", (int)decoded.len, (char *)decoded.data);

    // Cleanup.
    bytes_free(&decoded);
    tokvec_free(&tokens);
    tiktoken_free(enc);
}
```

### Getting a Vocabulary File

Download the cl100k_base vocabulary:

```bash
curl -o data/cl100k_base.tiktoken \
  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
```

### Building

```bash
# Prerequisites:
#   - GCC 14+ or Clang 18+ (for C23 support)
#   - PCRE2 (libpcre2-dev on Ubuntu, pcre2 via brew/vcpkg)
#   - CMake 3.25+

# Configure
cmake -B build -DCMAKE_C_COMPILER=gcc-14

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure

# Run examples
./build/count_tokens data/cl100k_base.tiktoken "Hello, world!"
./build/encode_decode data/cl100k_base.tiktoken "Hello, world!"
```

## C23 Feature: `#elifdef` / `#elifndef`

The last C23 feature we'll mention. These are shorthand for common
preprocessor patterns:

```c
// C17:
#ifdef __linux__
    // Linux
#elif defined(__APPLE__)
    // macOS
#elif defined(_WIN32)
    // Windows
#endif

// C23:
#ifdef __linux__
    // Linux
#elifdef __APPLE__
    // macOS
#elifdef _WIN32
    // Windows
#endif
```

Minor, but reduces visual noise in platform-detection code. Our
project doesn't have complex platform conditionals, so we just note
it here.

## Complete C23 Feature Reference

Here is every C23 feature we used or discussed, organized by category:

### Types and Declarations

| Feature | Chapter | Description |
|---------|---------|-------------|
| `bool` keyword | 1 | No `<stdbool.h>` needed |
| `nullptr` | 1 | Type-safe null pointer constant |
| `constexpr` | 1, 3 | Compile-time constant objects |
| `auto` | 4 | Local variable type inference |
| `enum : type` | 1, 7 | Fixed underlying type for enums |
| `typeof` | 2 | Standard type-of operator |
| `typeof_unqual` | 2 | Type-of stripping qualifiers |

### Attributes

| Feature | Chapter | Description |
|---------|---------|-------------|
| `[[nodiscard]]` | 1 | Warn on ignored return values |
| `[[maybe_unused]]` | 2 | Suppress unused-variable warnings |

### Assertions and Safety

| Feature | Chapter | Description |
|---------|---------|-------------|
| `static_assert(expr)` | 1 | No mandatory message string |
| `<stdckdint.h>` | 4 | Checked integer arithmetic |

### Standard Library

| Feature | Chapter | Description |
|---------|---------|-------------|
| `<stdbit.h>` | 3 | Bit manipulation functions |
| `strdup` / `strndup` | 6 | Now part of the C standard |

### Preprocessor

| Feature | Chapter | Description |
|---------|---------|-------------|
| `#elifdef` / `#elifndef` | 8 | Shorthand for `#elif defined(...)` |

### Initialization

| Feature | Chapter | Description |
|---------|---------|-------------|
| `= {}` empty initializer | 2 | Unambiguous zero-initialization |

## Architecture Recap

```
                    ┌──────────────────────┐
                    │   tiktoken.h         │  Convenience header
                    │   (Chapter 8)        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   encoding.h         │  Public API
                    │   (Chapter 7)        │  encode / decode / count
                    └──┬───────┬───────┬───┘
                       │       │       │
              ┌────────▼──┐ ┌──▼────┐ ┌▼────────┐
              │  regex.h  │ │ bpe.h │ │ vocab.h │
              │  (Ch. 5)  │ │(Ch.4) │ │ (Ch. 6) │
              └───────────┘ └───┬───┘ └──┬──┬───┘
                                │        │  │
                          ┌─────▼────────▼──▼───┐
                          │   hash.h + arena.h  │  Data structures
                          │     (Chapter 3)     │
                          └─────────┬───────────┘
                                    │
                          ┌─────────▼───────────┐
                          │     bytes.h         │  Fundamental type
                          │    (Chapter 2)      │
                          └─────────┬───────────┘
                                    │
                          ┌─────────▼───────────┐
                          │    base64.h         │  Data format
                          │    (Chapter 1)      │
                          └─────────────────────┘
```

Each layer depends only on the layers below it. No circular
dependencies, no forward declarations across modules.

## Design Decisions and Tradeoffs

### What We Did Well

1. **Arena allocation** — one free for 100K+ token entries. No
   individual malloc/free tracking needed for vocabulary data.

2. **Non-owning views** — the `cap == 0` convention lets us pass byte
   sequences to hash map lookups without copying.

3. **Opaque PCRE2 wrapper** — callers never include `pcre2.h`.
   The dependency is fully encapsulated.

4. **Layered architecture** — each module is testable in isolation.
   The test for Chapter 4 (BPE) doesn't need PCRE2 or file I/O.

### What We Could Improve

1. **The BPE loop is O(n²)** — fine for regex-split chunks (< 20
   bytes) but slow for long inputs. A priority queue would make it
   O(n log n).

2. **No error reporting** — functions return `nullptr` or empty
   results on failure, with no error message or code. A production
   library would use a result type with error details.

3. **Hash map duplication** — `B2iMap` and `I2bMap` are nearly
   identical. C23's `typeof` could generate both from a single macro
   template.

4. **Special token search is O(n×m)** — scanning for 5 tokens in
   long text. Aho-Corasick would be better for many special tokens.

5. **No streaming API** — you must provide the entire text upfront.
   A streaming encoder would be useful for large documents.

## What You've Learned

If you followed this tutorial from start to finish, you now know:

### About Tokenization
- How BPE (Byte Pair Encoding) works — both training and inference
- Why pre-tokenization with regex matters
- How tiktoken's vocabulary format is structured
- The role of special tokens in language model inputs

### About Modern C (C23)
- `constexpr` for compile-time constants (not just `const`)
- `nullptr` and `bool` as first-class keywords
- `[[nodiscard]]` and `[[maybe_unused]]` attributes
- `static_assert` without mandatory messages
- `typeof` for type-safe macros
- `enum : type` for ABI-stable enumerations
- `= {}` for unambiguous zero-initialization
- `<stdbit.h>` and `<stdckdint.h>` for safe operations

### About C Programming Patterns
- Arena allocators for bulk allocation with shared lifetimes
- Robin Hood hashing for cache-friendly hash maps
- Opaque types for encapsulation without classes
- Non-owning views (borrows) for zero-copy data passing
- The doubling growth strategy for amortized O(1) appends
- Branchless lookups via precomputed tables

## Going Further

Some directions to extend this project:

1. **Priority-queue BPE** — replace the O(n²) merge loop with a min-heap
   for O(n log n) performance.

2. **Streaming encoder** — process text incrementally without buffering
   the entire input.

3. **Thread safety** — make `TiktokenEncoding` safe to share across
   threads (the current implementation is not thread-safe because
   `Regex` reuses match data).

4. **Additional encodings** — implement `tiktoken_o200k_base()` and
   `tiktoken_p50k_base()` preset constructors.

5. **Benchmarking** — compare performance against Python tiktoken and
   the Rust implementation.

6. **WASM compilation** — compile to WebAssembly for browser-based
   token counting.

## Final Thoughts

C is sometimes dismissed as "too low-level" or "too dangerous" for
modern software. This tutorial hopefully demonstrates that modern C
(C23) is a capable, expressive language when used with care:

- **`constexpr`** brings compile-time safety previously only available
  in C++.
- **`nullptr`** and typed enums reduce classes of bugs.
- **Attributes** communicate intent to both compilers and humans.
- **Standard library additions** reduce dependence on platform-specific
  extensions.

The language is still fundamentally about explicit control — you manage
memory, you choose data layouts, you decide when and how to abstract.
That's not a limitation; it's the point. When you need to understand
exactly what your program does and why, C remains the right tool.
