"""Token counting with the embedding model's own tokenizer.

Chunk sizes are measured in the embedding model's tokens, not characters or words,
because that is the limit that matters: bge-small reads at most 512 tokens and
silently ignores anything after that.
"""

from collections.abc import Callable
from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache
def get_token_counter(model_name: str) -> Callable[[str], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # We only count tokens here, so silence the "sequence longer than 512" warning
    # that fires when measuring a long piece of text before it is split.
    tokenizer.model_max_length = 10**9

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
