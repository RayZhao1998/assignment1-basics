from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
import regex as re

pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# return vocab and merges
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize the vocabulary of [[...256 BYTE CHARS], <|endoftext|>]
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    next_index = 256
    for i, token in enumerate(special_tokens):
        assert token not in vocab.values(), f"Token {token} already in vocab"
        vocab[next_index] = token.encode("utf-8")
        next_index += 1

    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        content = f.read()
        # remove special_tokens and split into sentences
        sentences = re.split(b"|".join(re.escape(token).encode("utf-8") for token in special_tokens), content)

        # pre tokenization
        words_counts: dict[tuple[int, ...], int] = {}
        for sentence in sentences:
            matches = re.finditer(pre_tokenization_pattern, sentence.decode('utf-8'))
            for match in matches:
                word_bytes = tuple(match.group().encode("utf-8"))
                words_counts[word_bytes] = words_counts.get(word_bytes, 0) + 1

        # merge
        while next_index < vocab_size:
            counts = defaultdict(int)
            for word_bytes, count in words_counts.items():
                for i in range(len(word_bytes) - 1):
                    pair = (word_bytes[i], word_bytes[i + 1])
                    counts[pair] += count
            pair = max(counts, key=lambda x: (counts[x], (vocab[x[0]], vocab[x[1]])))
            index1, index2 = pair

            merges.append((vocab[index1], vocab[index2]))
            vocab[next_index] = vocab[index1] + vocab[index2]

            new_words_counts = {}
            for word_bytes, count in words_counts.items():
                new_word_bytes = []
                i = 0
                while i < len(word_bytes):
                    if i < len(word_bytes) - 1 and word_bytes[i] == index1 and word_bytes[i + 1] == index2:
                        new_word_bytes.append(next_index)
                        i += 2
                    else:
                        new_word_bytes.append(word_bytes[i])
                        i += 1
                new_words_counts[tuple(new_word_bytes)] = count
            words_counts = new_words_counts
            next_index += 1

    return vocab, merges

# train_bpe(
#     input_path="data/TinyStoriesV2-GPT4-valid.txt",
#     vocab_size=1000,
#     special_tokens=["<|endoftext|>"])
