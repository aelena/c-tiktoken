# Chapter 2 — Byte Strings

## The Problem With C Strings

C's native string representation — a null-terminated `char *` — has served
well for decades, but it's fundamentally unsuitable for a tokenizer:

1. **No embedded nulls.** The byte `0x00` is a valid byte in many
   encodings and can appear inside tokens. A C string terminates at the
   first `0x00`, silently truncating data.

2. **Length requires a scan.** `strlen()` walks the entire string to find
   the terminator. When you're processing millions of tokens, this adds up.

3. **No capacity tracking.** Growing a C string means calling `realloc`
   and hoping you remembered how much space you had.

4. **Signedness ambiguity.** `char` can be signed or unsigned depending on
   the platform. Byte data should be `uint8_t` — unambiguously an unsigned
   8-bit value.

We need a better type. Let's build one.

## The `Bytes` Type

```c
typedef struct {
    uint8_t *data;   // pointer to byte data
    size_t   len;    // number of valid bytes
    size_t   cap;    // allocated capacity (0 for non-owning views)
} Bytes;
```

This is sometimes called a "fat pointer" — it's a pointer plus metadata.
The three fields give us:

- **`data`**: the raw byte buffer
- **`len`**: how many bytes are in use (not including any terminator)
- **`cap`**: how many bytes are allocated (0 means we don't own the memory)

The `cap` field does double duty: it tracks available space for growth
*and* signals ownership. A `Bytes` with `cap > 0` owns its memory and
must be freed. A `Bytes` with `cap == 0` is a non-owning **view** (a
slice) and must not be freed.

## C23 Feature: Empty Initializer `= {}`

How do you zero-initialize a struct in C? In C17:

```c
Bytes b = {0};       // Works, but initializes first member with 0
Bytes b = {NULL};    // More explicit, but only initializes first member
```

Both of these technically only initialize the *first* member; remaining
members are implicitly zero-initialized by the language rules. This works,
but the intent is unclear — are you deliberately setting `data = 0` (an
integer), or are you trying to zero the whole struct?

C23 introduces the empty initializer:

```c
Bytes b = {};    // Every member is zero-initialized. Unambiguous.
```

`= {}` value-initializes every member. Pointers become null pointers
(not integer 0), arithmetic types become 0, nested structs are recursively
zero-initialized. It's the C23 idiom for "give me a zeroed-out value."

We use this throughout:

```c
Bytes bytes_with_cap(size_t cap) {
    if (cap == 0) {
        return (Bytes){};     // return a zeroed Bytes
    }
    // ...
}
```

## C23 Feature: `typeof`

`typeof` has been a compiler extension (GCC and Clang) for decades. C23
makes it official:

```c
typeof(x) y = x;    // y has the same type as x
```

This is most powerful in macros. Consider a generic "swap" macro:

```c
// C17: not possible to do safely without knowing the type.
// C23:
#define SWAP(a, b) do {         \
    typeof(a) tmp_ = (a);       \
    (a) = (b);                  \
    (b) = tmp_;                 \
} while (0)
```

The `typeof(a)` expression evaluates to the type of `a` without
evaluating `a` itself. No side effects, no multiple evaluation.

C23 also adds `typeof_unqual`, which strips qualifiers like `const` and
`volatile`:

```c
const int x = 42;
typeof(x)        y = 10;   // y is const int — can't modify it
typeof_unqual(x) z = 10;   // z is int — mutable
```

We'll use both extensively in Chapter 3 (type-generic hash map macros).
For now, `typeof` appears in our internal growth helpers.

## C23 Feature: `[[maybe_unused]]`

When a variable or parameter is intentionally unused, most compilers warn.
In C17, the common workaround was:

```c
(void)unused_param;     // Cast to void to suppress warning
```

C23 provides a standard attribute:

```c
[[maybe_unused]]
static constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
```

This tells the compiler: "I know this might not be used in every
configuration — that's intentional, not a bug." We use it on constants
that are referenced in one function but that we want to keep visible
and documented at file scope.

## Dynamic Arrays: The Growth Pattern

Both `Bytes`, `ByteVec`, and `TokenVec` use the same growth strategy:

```c
static inline size_t grow_cap(size_t current, size_t needed) {
    size_t cap = current ? current : 16;
    while (cap < needed) {
        cap *= 2;
    }
    return cap;
}
```

This is the **doubling strategy** — when you run out of space, double the
capacity. The math works out nicely:

- After `n` appends, you've done at most `log₂(n)` reallocations.
- Each reallocation copies at most `n` bytes.
- Total copy cost across all `n` appends: O(n).
- **Amortized cost per append: O(1).**

The minimum capacity of 16 avoids tiny allocations. Starting at 1 and
doubling would mean your first 4 appends each trigger a reallocation
(1→2→4→8). Starting at 16 gives you 16 appends before the first growth.

## Ownership and Views

One of the trickiest parts of C programming is tracking who owns allocated
memory. Our convention:

| `cap` value | Meaning | Must free? |
|-------------|---------|------------|
| `cap > 0` | Owning — this `Bytes` allocated the buffer | Yes |
| `cap == 0` | Non-owning view (slice) | No |

The `bytes_slice` function creates a view:

```c
Bytes bytes_slice(Bytes b, size_t start, size_t end) {
    // ...
    return (Bytes){
        .data = b.data + start,
        .len  = end - start,
        .cap  = 0,       // I don't own this memory!
    };
}
```

And `bytes_free` checks ownership before freeing:

```c
void bytes_free(Bytes *b) {
    if (b != nullptr && b->cap > 0) {
        free(b->data);
    }
    if (b != nullptr) {
        *b = (Bytes){};
    }
}
```

This is a simple form of the **borrow** pattern. A slice "borrows" memory
from an owning `Bytes`. The slice is valid only as long as the owner lives.
The compiler won't enforce this for us (that's Rust's territory), but the
convention makes it manageable.

## FNV-1a Hashing

We need a hash function for `Bytes` because in Chapter 3, byte sequences
will be hash map keys. We use **FNV-1a** (Fowler-Noll-Vo, variant 1a):

```c
uint64_t bytes_hash(Bytes b) {
    uint64_t hash = 0xcbf29ce484222325ULL;    // offset basis
    for (size_t i = 0; i < b.len; i++) {
        hash ^= (uint64_t)b.data[i];         // XOR byte in
        hash *= 0x00000100000001B3ULL;        // multiply by prime
    }
    return hash;
}
```

FNV-1a has these properties:
- **Simple** — the entire algorithm is two operations per byte.
- **Fast** — no complex math, just XOR and multiply.
- **Well-distributed** — good avalanche behavior for typical inputs.
- **Deterministic** — same input always produces same output.

It's not cryptographic (don't use it for security), but it's excellent for
hash tables.

**Why XOR-then-multiply (1a) instead of multiply-then-XOR (1)?** The 1a
variant has better avalanche properties — changing one input bit affects
more output bits. The improvement is small but free.

## The Dynamic Array Types

We define two more dynamic arrays besides `Bytes`:

### `ByteVec` — a vector of `Bytes` values

```c
typedef struct {
    Bytes  *items;
    size_t  len;
    size_t  cap;
} ByteVec;
```

Used for: collecting regex match results, storing BPE parts during the
merge algorithm, building lists of tokens as byte sequences.

### `TokenVec` — a vector of `uint32_t` token IDs

```c
typedef struct {
    uint32_t *items;
    size_t    len;
    size_t    cap;
} TokenVec;
```

Used for: the output of encoding (text → token IDs) and the input of
decoding (token IDs → text).

Both follow the same growth pattern as `Bytes`. The only difference is the
element type and size.

## Why Not Use a Generic Container?

You might wonder why we have three nearly identical dynamic arrays
(`Bytes`, `ByteVec`, `TokenVec`) instead of one generic container. In C++,
you'd use `std::vector<T>`. In C, the options are:

1. **`void *` arrays** — type-unsafe, require casts everywhere, lose
   size information.
2. **Macro-generated types** — works (and we'll use this for hash maps in
   Chapter 3), but adds complexity.
3. **Separate concrete types** — a bit repetitive, but simple, type-safe,
   and easy to debug.

For a tutorial, clarity wins over DRY (Don't Repeat Yourself). Three
small, concrete types are easier to understand than one macro-generated
generic. We'll go generic for the hash map where the benefit is clearer.

## Building and Running

Add to `CMakeLists.txt`:

```cmake
# Add to the library sources
add_library(tiktoken
    src/base64.c
    src/bytes.c        # NEW
)

# Add test
add_executable(test_bytes tests/test_bytes.c)
target_link_libraries(test_bytes PRIVATE tiktoken)
add_test(NAME bytes COMMAND test_bytes)
```

## What's Next

We now have two foundational types:
- A base64 decoder that turns vocabulary data into raw bytes
- A `Bytes` type that holds those bytes safely

The next piece is **storage** — we need a way to look up tokens by their
byte content (for encoding) and by their rank (for decoding). That means
we need a hash map.

In [Chapter 3](chapter03_hashmap.md), we'll build a Robin Hood hash map
and an arena allocator, using `typeof` to make the map work with different
key and value types.

## Summary of C23 Features Introduced

| Feature | What It Replaces | Why It's Better |
|---------|-----------------|-----------------|
| `= {}` empty initializer | `= {0}` / `= {NULL}` | Unambiguous zero-init for any type |
| `typeof` (standard) | GCC/Clang `__typeof__` extension | Portable, official, no compiler flags |
| `typeof_unqual` | No equivalent | Strips const/volatile for mutable copies |
| `[[maybe_unused]]` | `(void)x` cast | Standard, self-documenting intent |
