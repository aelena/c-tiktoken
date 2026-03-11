# Chapter 4 — The BPE Algorithm

## What Is Byte Pair Encoding?

BPE (Byte Pair Encoding) is a compression algorithm repurposed for
tokenization. The idea is simple:

1. Start with a vocabulary of individual bytes (256 tokens).
2. Find the most frequent pair of adjacent tokens in a training corpus.
3. Merge that pair into a new token and add it to the vocabulary.
4. Repeat until the vocabulary reaches the desired size.

Steps 1–4 happen **once**, during training. OpenAI has already done this
and published the resulting vocabulary files. Our job is the **inference**
side: given a trained vocabulary (a set of merge rules with priority
rankings), apply those merges to new text.

## The Merge Algorithm

Given input bytes and a vocabulary of ranked merges, encoding works as
follows:

### Step 1: Initialize

Start with each input byte as its own "part":

```
Input: "hello"
Parts: [h] [e] [l] [l] [o]
```

### Step 2: Find the best merge

For every adjacent pair of parts, concatenate them and look up the
result in the vocabulary. The pair with the **lowest rank** (highest
priority) wins.

```
Pairs:  he(300)  el(none)  ll(301)  lo(302)
Best:   he(300) — lowest rank
```

### Step 3: Apply the merge

Replace the winning pair with a single merged part:

```
Parts: [he] [l] [l] [o]
```

### Step 4: Repeat

Go back to step 2 with the new parts. Continue until no more merges
are possible.

```
Iteration 2:  [he] [l] [l] [o]   →  ll(301) wins  →  [he] [ll] [o]
Iteration 3:  [he] [ll] [o]      →  hell(304) wins →  [hell] [o]
Iteration 4:  [hell] [o]         →  hello(305) wins → [hello]
Done!
```

### Step 5: Collect token IDs

Each remaining part maps to a token ID (its rank in the vocabulary):

```
[hello] → rank 305 → token ID 305
```

## Data Structure: The Indexed Linked List

The naive approach — rebuild an array after each merge — is O(n) per
merge and O(n²) total. We use an **indexed linked list** instead:

```c
typedef struct {
    size_t   start;    // start index in input bytes
    size_t   end;      // end index in input bytes (exclusive)
    int32_t  next;     // index of next part (-1 = end of list)
    uint32_t rank;     // merge rank of (this ++ next), or NO_RANK
} Part;
```

Each part is a byte range `[start, end)` in the original input. Merging
part `i` with part `next(i)` just extends `i`'s byte range and removes
`next(i)` from the list — O(1) per merge.

The tradeoff: finding the minimum-rank pair is still O(n) per iteration
(a linear scan), making the overall algorithm O(n²). For tiktoken's
typical inputs (words from regex splitting, usually under 20 bytes),
this is fast enough.

### Why Not a Priority Queue?

tiktoken's Rust implementation uses a priority queue for longer inputs,
reducing the complexity to O(n log n). The priority queue holds all
pairs keyed by their merge rank. After each merge, you update the
affected pairs (the pair to the left and right of the merge point).

For our tutorial, the O(n²) approach is clearer and correct. Here's
when you'd switch to a priority queue:

- Inputs regularly exceed ~100 bytes (rare after regex splitting)
- You're tokenizing without pre-tokenization (the entire input at once)
- You need to handle pathological inputs (e.g., 10KB of single chars
  with many possible merges)

The priority queue version is a worthwhile exercise but adds complexity
(heap operations, invalidation tracking) that would obscure the core
algorithm.

## The get_rank Function

The most called function in the entire algorithm. It checks whether a
byte range has a merge rank:

```c
static uint32_t get_rank(const BpeRanks *ranks,
                         const uint8_t *data,
                         size_t start, size_t end)
{
    Bytes key = {
        .data = (uint8_t *)(data + start),
        .len  = end - start,
        .cap  = 0,   // non-owning view — no allocation!
    };
    uint32_t rank = 0;
    if (b2i_get(&ranks->encoder, key, &rank)) {
        return rank;
    }
    return NO_RANK;
}
```

Key design choices:

1. **Zero-copy lookup.** We create a non-owning `Bytes` view into the
   original input. No `malloc`, no `memcpy`. The hash map lookup uses
   this view to hash and compare against stored keys.

2. **Sentinel return value.** `NO_RANK = UINT32_MAX` signals "no merge
   exists." Real ranks are always smaller (tiktoken vocabularies have
   at most ~200K entries).

3. **The `cap = 0` trick.** We use the ownership convention from Chapter
   2: `cap == 0` means this `Bytes` doesn't own its memory. This lets
   us pass it to `b2i_get` (which only reads the key) without worrying
   about who frees it.

## Decoding: The Easy Direction

Decoding is trivial — just concatenate the byte sequence for each token:

```c
Bytes bpe_decode(const BpeRanks *ranks, const uint32_t *tokens, size_t n) {
    Bytes result = {};
    for (size_t i = 0; i < n; i++) {
        Bytes token_bytes = {};
        if (i2b_get(&ranks->decoder, tokens[i], &token_bytes)) {
            bytes_append(&result, token_bytes.data, token_bytes.len);
        }
    }
    return result;
}
```

No merge algorithm needed — the vocabulary maps each rank directly to
its byte sequence. The `i2b_get` returns a non-owning view into the
hash map's stored data, and `bytes_append` copies those bytes into
the result.

## C23 Feature: `auto` Type Inference

C23 allows `auto` for local variables:

```c
auto x = 42;          // x is int
auto p = &buffer[0];  // p is uint8_t*
```

This is more limited than C++ `auto`:
- Only for local variables with initializers
- Cannot be used for function parameters or return types
- The type must be unambiguously deducible

We use it sparingly — explicit types are usually clearer in C. It's
most useful for long type names like iterators (if we had them) or
when the type is obvious from the right-hand side.

## C23 Feature: `<stdckdint.h>` — Checked Integer Arithmetic

C23 adds `<stdckdint.h>` for overflow-safe arithmetic:

```c
#include <stdckdint.h>

size_t a = SIZE_MAX;
size_t b = 1;
size_t result;

if (ckd_add(&result, a, b)) {
    // overflow detected!
}
```

The `ckd_add`, `ckd_sub`, and `ckd_mul` functions perform the operation
and return `true` if overflow occurred. This replaces fragile manual
checks like:

```c
// C17 overflow check — easy to get wrong
if (a > SIZE_MAX - b) { /* overflow */ }
```

In our BPE code, we could use `ckd_add` when computing buffer sizes.
For the tutorial, we note it as a best practice but don't add it to
every arithmetic operation — that would obscure the algorithm.

## Performance Analysis

For a word of length `n`:

| Operation | Cost per merge | Total cost |
|-----------|---------------|------------|
| Find min rank | O(n) scan | O(n) × O(n) merges = O(n²) |
| Merge parts | O(1) linked list ops | O(n) total |
| Recompute neighbor ranks | O(1) hash lookups | O(n) total |
| Collect results | O(n) final scan | O(n) |
| **Total** | | **O(n²)** |

For typical tiktoken inputs after regex splitting, `n` is usually:
- English words: 3–15 bytes
- Numbers: 1–3 digits
- Punctuation: 1–5 bytes
- Whitespace: 1–3 bytes

With n ≤ 20, even O(n²) is effectively instant. The real performance
bottleneck in a production tokenizer is the regex splitting (Chapter 5)
and hash map lookups, not the merge loop.

## Testing With a Toy Vocabulary

Our tests use a hand-crafted vocabulary for "hello":

```
'h'     → 104    (single byte)
'e'     → 101
'l'     → 108
'o'     → 111
"he"    → 300    (merge rank: applied first)
"ll"    → 301
"lo"    → 302
"hel"   → 303
"hell"  → 304
"hello" → 305    (merge rank: applied last)
```

This lets us verify the exact merge sequence without needing a real
vocabulary file. The test traces through each merge step and checks
that the final token IDs match expectations.

## Building and Running

Updated `CMakeLists.txt`:

```cmake
add_library(tiktoken
    src/base64.c
    src/bytes.c
    src/arena.c
    src/hash.c
    src/bpe.c         # NEW
)

add_executable(test_bpe tests/test_bpe.c)
target_link_libraries(test_bpe PRIVATE tiktoken)
add_test(NAME bpe COMMAND test_bpe)
```

## What's Next

The BPE algorithm operates on individual byte sequences — but tiktoken
doesn't feed the entire input text to BPE at once. First, it splits the
text into chunks using a regex pattern. Each chunk is then independently
encoded with BPE.

In [Chapter 5](chapter05_regex.md), we'll implement the regex
pre-tokenization step using PCRE2, handling Unicode-aware patterns like
`\p{L}` (letters) and `\p{N}` (numbers).

## Summary of C23 Features Introduced

| Feature | What It Replaces | Why It's Better |
|---------|-----------------|-----------------|
| `auto` | Explicit type on every variable | Less repetition for obvious types |
| `<stdckdint.h>` | Manual overflow checks | Correct, readable, hard to misuse |
