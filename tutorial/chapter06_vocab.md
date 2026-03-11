# Chapter 6 — Vocabulary Loading

## The .tiktoken File Format

OpenAI distributes tiktoken vocabularies as plain text files with a
simple format. Each line contains a base64-encoded token and its rank:

```
IQ== 0
Ig== 1
Iw== 2
JA== 3
JQ== 4
...
```

The cl100k_base vocabulary (used by GPT-4) has 100,256 entries — about
1.6 MB of text. Let's decode what these lines mean:

```
IQ==  →  base64_decode("IQ==")  →  bytes [0x21]  →  "!"  →  rank 0
Ig==  →  base64_decode("Ig==")  →  bytes [0x22]  →  '"'  →  rank 1
Iw==  →  base64_decode("Iw==")  →  bytes [0x23]  →  "#"  →  rank 2
```

The first 256 entries typically correspond to single bytes (0x00–0xFF),
though not necessarily in order. Higher-ranked entries are multi-byte
sequences formed by BPE merges during training.

## The Loading Pipeline

Loading a vocabulary file ties together four modules we've already built:

```
File on disk
    │
    ▼  fread (read entire file into memory)
Memory buffer
    │
    ▼  Line-by-line parsing
Each line: "YQ== 0"
    │
    ├─▶ Left side: "YQ=="  ──▶  base64_decode (Chapter 1)  ──▶  bytes [0x61]
    │
    └─▶ Right side: "0"    ──▶  strtoul                     ──▶  rank 0
    │
    ▼  Store
    ├─▶ arena_push_bytes (Chapter 3)     — copy bytes into arena
    ├─▶ b2i_insert (Chapter 3)           — bytes → rank map
    └─▶ i2b_insert (Chapter 3)           — rank → bytes map
```

## Pre-Sizing for Performance

Loading 100K entries into a hash map that starts at capacity 16 means
many reallocations and rehashes. We avoid this by pre-sizing:

```c
size_t estimated_entries = count_lines(data, data_len);

result.arena  = arena_new(estimated_entries * 20);        // ~2 MB
result.ranks.encoder = b2i_new(estimated_entries * 2);    // 200K slots
result.ranks.decoder = i2b_new(estimated_entries * 2);    // 200K slots
```

The `count_lines` scan is O(n) over the file data — trivially fast
compared to the actual loading work. By starting with capacity for
2× the expected entries, the hash maps won't need to grow at all
(their load factor stays under 50%, well below the 70% threshold).

The arena is sized at 20 bytes per entry. Most tokens are 1–8 bytes,
but some Unicode tokens can be longer. 20 bytes per entry is generous
enough that the arena rarely needs to grow.

## Memory Ownership

This is where the ownership model from Chapters 2 and 3 pays off:

```
VocabResult
├── arena: Arena        ← owns all token byte data
├── ranks.encoder: B2iMap  ← keys are non-owning views into arena
└── ranks.decoder: I2bMap  ← values are non-owning views into arena
```

When we parse a line:

1. Base64-decode into a temporary buffer (`malloc` / `free`).
2. Copy the decoded bytes into the arena (`arena_push_bytes`).
3. Create a non-owning `Bytes` view into the arena data (`cap = 0`).
4. Insert this view as a hash map key (B2iMap) or value (I2bMap).

The hash map entries point into the arena but don't own the data.
When we free everything, we free the arena once — no need to iterate
100K entries and free each key individually.

```c
void vocab_free(VocabResult *v) {
    b2i_free(&v->ranks.encoder);    // frees the slot arrays
    i2b_free(&v->ranks.decoder);    // frees the slot arrays
    arena_free(&v->arena);          // frees ALL token byte data at once
}
```

This is the arena allocator pattern in action: **bulk allocation with
a shared lifetime.**

## Parsing Details

### Finding the separator

```c
const char *space = nullptr;
for (size_t i = 0; i < line_len; i++) {
    if (line[i] == ' ') {
        space = &line[i];
        break;
    }
}
```

We find the first space character to split the line. This is simpler
and safer than `strtok` (which modifies the input string and uses
hidden global state).

### Parsing the rank

```c
char rank_buf[32];
memcpy(rank_buf, rank_str, rank_len);
rank_buf[rank_len] = '\0';

errno = 0;
unsigned long rank_val = strtoul(rank_buf, nullptr, 10);
if (errno != 0 || rank_val > UINT32_MAX) return false;
```

We copy the rank substring into a small stack buffer (ranks are at most
6 digits, so 32 bytes is plenty) and use `strtoul` to parse it.

**C23 note:** `<stdckdint.h>` provides `ckd_mul`, `ckd_add`, etc. for
checked arithmetic. We could build a safe integer parser using these,
but `strtoul` + `errno` is the established C idiom and works well here.

### Error tolerance

Malformed lines (no space separator, invalid base64, rank overflow) are
silently skipped. This makes the loader robust against minor format
variations without crashing:

```c
if (parse_line(cursor, line_len, &result.arena,
               &token_bytes, &rank)) {
    // success — insert into maps
} else {
    // skip this line
}
```

## C23 Feature: `strdup` and `strndup` in the Standard

C23 officially adds `strdup()` and `strndup()` to `<string.h>`. These
have been POSIX extensions since forever, but were never formally part
of the C standard. This matters for strict conformance and for platforms
that don't implement POSIX.

```c
// Previously: POSIX only, or needed _POSIX_C_SOURCE
// C23: standard, available everywhere
char *copy = strdup("hello");
char *partial = strndup("hello world", 5);  // "hello"
```

We don't use them in this chapter (our parsing doesn't need string
duplication), but they're worth knowing about — they eliminate one
of the most common "roll your own" functions in C codebases.

## Testing With Synthetic Data

Our tests use a tiny in-memory vocabulary:

```c
static const char TINY_VOCAB[] =
    "YQ== 0\n"     // "a" → rank 0
    "Yg== 1\n"     // "b" → rank 1
    "Yw== 2\n"     // "c" → rank 2
    "YWI= 3\n"     // "ab" → rank 3
    "YmM= 4\n"     // "bc" → rank 4
    "YWJj 5\n";    // "abc" → rank 5
```

This is enough to verify:
- Correct base64 decoding of tokens
- Both encode and decode direction lookups
- BPE encoding with the loaded vocabulary
- Encode→decode roundtrip
- Error handling (empty input, null, malformed lines)

For production validation, you'd compare against Python tiktoken's
output for the same vocabulary file.

## Building

Updated `CMakeLists.txt`:

```cmake
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c
    src/hash.c
    src/bpe.c
    src/regex.c
    src/vocab.c        # NEW
)

add_executable(test_vocab tests/test_vocab.c)
target_link_libraries(test_vocab PRIVATE tiktoken)
add_test(NAME vocab COMMAND test_vocab)
```

## What's Next

We now have every piece needed for a complete tokenizer:
- Base64 decoding (Chapter 1)
- Byte strings (Chapter 2)
- Hash maps and arena (Chapter 3)
- BPE algorithm (Chapter 4)
- Regex pre-tokenization (Chapter 5)
- Vocabulary loading (Chapter 6)

In [Chapter 7](chapter07_api.md), we'll combine everything into the
public `TiktokenEncoding` API — `tiktoken_encode()`, `tiktoken_decode()`,
with special token handling and preset encoding constructors.

## Summary of C23 Features Discussed

| Feature | Context |
|---------|---------|
| `strdup` / `strndup` | Now standard in C23 (previously POSIX only) |
| `<stdckdint.h>` | Alternative to errno-based overflow checking |
| `= {}` empty init | Zero-initializing VocabResult on error paths |
| Non-owning views | `cap = 0` convention for arena-backed Bytes |
