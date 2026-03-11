#!/usr/bin/env python3
"""
Reference implementation using official tiktoken library.

This script encodes text using the official tiktoken library and outputs
the token IDs as JSON. Used by C integration tests to validate correctness.
"""

import json
import sys
import tiktoken


def encode_text(encoding_name: str, text: str, allow_special: bool = False) -> list[int]:
    """
    Encode text using the official tiktoken library.
    
    Args:
        encoding_name: Name of the encoding (e.g., "cl100k_base")
        text: Text to encode
        allow_special: If True, allow special tokens; if False, treat as ordinary text
    
    Returns:
        List of token IDs
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
        if allow_special:
            tokens = enc.encode(text, allowed_special="all")
        else:
            tokens = enc.encode(text, allowed_special=set())
        return tokens
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: tiktoken_reference.py <encoding_name> <text> [allow_special]", file=sys.stderr)
        sys.exit(1)
    
    encoding_name = sys.argv[1]
    text = sys.argv[2]
    allow_special = len(sys.argv) > 3 and sys.argv[3] == "1"
    
    tokens = encode_text(encoding_name, text, allow_special)
    
    # Output as JSON array
    print(json.dumps(tokens))


if __name__ == "__main__":
    main()
