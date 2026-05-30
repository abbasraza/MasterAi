
# this file is for testing tiktoken and understanding how tokenization works for different models.
# It is not part of the main codebase, but can be used as a reference for how to use tiktoken to
# count tokens and understand tokenization.

import tiktoken

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

def print_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    tokens = encoding.encode(string)
    for i, token in enumerate(tokens):
        print(f"Token {i}: {token} -> {encoding.decode([token])}")
    return num_tokens

print_tokens_from_string(question, "cl100k_base")


