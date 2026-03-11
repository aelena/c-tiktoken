# Chapter 1 — Base64 Decoding

## Why Base64?

Before we can build a tokenizer, we need to read its vocabulary. OpenAI's
tiktoken stores vocabulary files in a simple text format where each line
looks like this:

```
IQ==  0
Ig==  1
Iw==  2
...
```

The left column is a **base64-encoded byte sequence** — the raw bytes that
make up a token. The right column is the token's **rank** (its integer ID).

Base64 is the universal way to embed arbitrary binary data in text. Since
tokens can be *any* byte sequence (including bytes that aren't valid ASCII
or UTF-8), base64 gives us a safe, portable encoding. Our first task is to
build a decoder.

## What Is Base64?

Base64 encodes binary data using 64 printable ASCII characters:

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z   (0–25)
a b c d e f g h i j k l m n o p q r s t u v w x y z   (26–51)
0 1 2 3 4 5 6 7 8 9                                     (52–61)
+ /                                                      (62–63)
```

Each character carries 6 bits of information. Four characters (4 × 6 = 24
bits) encode exactly three bytes (3 × 8 = 24 bits):

```
  Input bytes:    [  byte 0  ] [  byte 1  ] [  byte 2  ]
  Bits:           xxxxxxxx     yyyyyyyy     zzzzzzzzzz
  Regrouped:      [xxxxxx] [xxyyyy] [yyyyzz] [zzzzzz]
  Base64 chars:     A–/       A–/       A–/       A–/
```

If the input length isn't a multiple of 3, the last group is shorter:
- **2 bytes** → 3 base64 characters + one `=` pad
- **1 byte**  → 2 base64 characters + two `==` pads

## The Lookup Table Approach

The naïve way to decode a base64 character is a chain of `if/else`:

```c
// Don't do this.
if (c >= 'A' && c <= 'Z') return c - 'A';
else if (c >= 'a' && c <= 'z') return c - 'a' + 26;
else if (c >= '0' && c <= '9') return c - '0' + 52;
else if (c == '+') return 62;
else if (c == '/') return 63;
else return INVALID;
```

That's 5 branches per character. With a 256-entry lookup table, decoding
becomes a single array access — branchless and cache-friendly:

```c
uint8_t value = DECODE_TABLE[(unsigned char)c];
```

This is a classic C technique: **trade a small amount of memory (256 bytes)
for the elimination of all branching in a hot loop.**

## C23 Feature: `constexpr`

In C17, you'd declare the table as:

```c
static const uint8_t DECODE_TABLE[256] = { ... };
```

`const` means "I won't modify this" — but it doesn't guarantee the compiler
evaluates it at compile time. In practice, `static const` arrays *are*
placed in read-only data sections, but the language doesn't require it.

C23 introduces `constexpr`:

```c
static constexpr uint8_t DECODE_TABLE[256] = { ... };
```

This is a **compile-time guarantee**. The compiler *must* be able to evaluate
every element of the initializer at compile time. If you accidentally write
an initializer that depends on a runtime value, the compiler will reject it.

### Rules for `constexpr` in C23

- Only applies to **objects** (variables), not functions. (C++ has
  `constexpr` functions; C23 does not.)
- The initializer must be a constant expression.
- The object has no address at runtime if the compiler can prove it's never
  needed (the optimizer may inline all uses).
- You can use `constexpr` objects as array sizes, `case` labels, and in
  other contexts requiring integer constant expressions.

**Example: using `constexpr` for the sentinel value:**

```c
constexpr uint8_t B64_INV = 0xFF;
```

This is cleaner than `#define B64_INV 0xFF` because it's typed (it's a
`uint8_t`, not just a preprocessor substitution), and it respects scope.

## C23 Feature: `static_assert` Without a Message

C11 introduced `_Static_assert(expr, "message")` (and the `static_assert`
convenience macro via `<assert.h>`). The message string was mandatory.

C23 makes the message **optional** and promotes `static_assert` to a
keyword:

```c
// C11 way:
_Static_assert(sizeof(DECODE_TABLE) == 256, "table must cover all bytes");

// C23 way:
static_assert(sizeof(DECODE_TABLE) == 256);
```

We use this to verify our table has exactly 256 entries. If someone
accidentally adds or removes a line, the build fails immediately — not at
runtime.

## C23 Feature: Fixed-Width Enums

Traditional C enums have an implementation-defined underlying type (usually
`int`, but the standard doesn't guarantee this). C23 lets you be explicit:

```c
enum b64_status : int {
    B64_OK             =  0,
    B64_INVALID_CHAR   = -1,
    B64_INVALID_LENGTH = -2,
};
```

The `: int` suffix fixes the underlying type. Benefits:
- **ABI stability** — the size and signedness are part of the declaration,
  not up to the compiler.
- **Self-documenting** — readers know the enum fits in an `int`.
- **Interop** — when passing enums across library boundaries or storing
  them in structures, the size is guaranteed.

## C23 Feature: `[[nodiscard]]`

The `[[nodiscard]]` attribute causes a compiler warning when the return
value is silently ignored:

```c
[[nodiscard]]
enum b64_status b64_decode(const char *input, size_t input_len,
                           uint8_t *output, size_t *output_len);
```

Calling `b64_decode(...)` without checking the return value will now
trigger a warning. This is exactly right for error codes — ignoring a
decode failure is almost certainly a bug.

In C17 you could achieve something similar with compiler-specific
`__attribute__((warn_unused_result))`, but `[[nodiscard]]` is the
standard, portable way.

## C23 Feature: `bool`, `true`, `false` as Keywords

In C17:
```c
#include <stdbool.h>  // defines bool, true, false as macros
bool ok = true;
```

In C23:
```c
// No include needed — bool, true, false are keywords.
bool ok = true;
```

This is a small change, but it eliminates a frequent source of confusion
and one more `#include` to remember.

## C23 Feature: `nullptr`

C17's `NULL` is typically `((void *)0)` or just `0`. It has an integer
type in some implementations, which causes subtle bugs:

```c
// C17: this compiles without warning on some compilers.
int x = NULL;  // Probably not what you wanted.
```

C23 introduces `nullptr`, which has type `nullptr_t` and converts *only*
to pointer types:

```c
int x = nullptr;   // Compiler error — nullptr is not an integer.
int *p = nullptr;   // OK — nullptr converts to any pointer type.
```

Throughout our codebase, we use `nullptr` instead of `NULL` for null
pointer checks.

## The Decoding Algorithm

Here's the full decode loop, step by step:

### Step 1: Strip padding and whitespace

Tiktoken files use standard padded base64, but we strip `=`, `\n`, `\r`,
and spaces from the end so we can handle both padded and unpadded input.

### Step 2: Compute group sizes

```c
size_t full_quads = input_len / 4;   // complete 4-char groups
size_t remainder  = input_len % 4;   // 0, 2, or 3 trailing chars
```

A remainder of 1 is invalid (6 bits < 8 bits — can't produce a byte).

### Step 3: Decode full groups

For each group of 4 characters, look up their 6-bit values, pack them into
a 24-bit integer, and extract 3 bytes:

```c
uint32_t triple = ((uint32_t)a << 18)
                | ((uint32_t)b << 12)
                | ((uint32_t)c <<  6)
                | ((uint32_t)d <<  0);

output[out_pos++] = (uint8_t)(triple >> 16);  // bits 23–16
output[out_pos++] = (uint8_t)(triple >>  8);  // bits 15–8
output[out_pos++] = (uint8_t)(triple >>  0);  // bits  7–0
```

### Step 4: Decode the remainder

- **2 trailing chars** → 12 bits → 1 byte (top 8 bits)
- **3 trailing chars** → 18 bits → 2 bytes (top 16 bits)

## API Design Decisions

### Caller-provided output buffer

```c
enum b64_status b64_decode(const char *input, size_t input_len,
                           uint8_t *output, size_t *output_len);
```

The caller allocates the output buffer and passes its size implicitly via
`b64_decoded_size()`. This is the idiomatic C approach:

- **No hidden allocations.** The function doesn't call `malloc`, so there's
  no memory ownership ambiguity.
- **The caller controls the allocator.** In later chapters we'll use an
  arena allocator — this function doesn't need to know about that.
- **Composable.** You can decode into a stack buffer, a heap buffer, or
  a subregion of a larger allocation.

### Separate sizing function

`b64_decoded_size()` returns an upper bound. The actual decoded size might
be smaller (because padding characters don't contribute to output). The
exact size comes back through the `*output_len` out-parameter.

## Testing

Our test file `tests/test_base64.c` includes:

1. **RFC 4648 vectors** — the standard test cases for base64.
2. **Unpadded input** — since we strip padding, "Zg" should decode the
   same as "Zg==".
3. **Binary data** — non-ASCII bytes like `0xDEADBEEF` to verify we
   handle the full byte range.
4. **Error cases** — invalid characters, invalid lengths, null pointers.
5. **Tiktoken-like data** — strings representative of real vocabulary
   entries, including multi-byte UTF-8.

## Building and Running

```bash
# Configure (requires GCC 14+ or Clang 18+ for C23 support)
cmake -B build -DCMAKE_C_COMPILER=gcc-14

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure
```

## What's Next

With base64 decoding in hand, we can extract raw bytes from tiktoken's
vocabulary files. But those bytes need a home — we need a data type that
represents "an arbitrary sequence of bytes with a known length." That's
not a C string (which is null-terminated and can't contain `\0` bytes).

In [Chapter 2](chapter02_bytestrings.md), we'll build `Bytes` — a
length-prefixed dynamic byte buffer that will be the fundamental data
type throughout the rest of the project.

## Summary of C23 Features Introduced

| Feature | What It Replaces | Why It's Better |
|---------|-----------------|-----------------|
| `constexpr` | `static const` / `#define` | Compile-time guarantee, typed, scoped |
| `static_assert(expr)` | `_Static_assert(expr, "msg")` | No mandatory message, is a keyword |
| `enum name : type` | `enum name` | Fixed underlying type, ABI-stable |
| `[[nodiscard]]` | `__attribute__((warn_unused_result))` | Standard, portable |
| `bool` keyword | `#include <stdbool.h>` | Built-in, no header needed |
| `nullptr` | `NULL` / `(void *)0` | Type-safe, won't convert to integer |
