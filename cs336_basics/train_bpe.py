from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
import regex as re
from functools import partial
from tqdm.contrib.concurrent import process_map
import time

pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
num_processes = 4

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
    for token in special_tokens:
        vocab[next_index] = token.encode("utf-8")
        next_index += 1

    #  --------------------------------
    # | Parallelizing Pre tokenziation |
    #  --------------------------------
    parallel_pre_tokenziation_start = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

    if chunk_pairs:
        starts, ends = zip(*chunk_pairs)
        chunk_counts = process_map(
            partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
            starts,
            ends,
            max_workers=num_processes,
            chunksize=1,
        )
    else:
        chunk_counts = []

    words_counts = defaultdict(int)

    for chunk_count in chunk_counts:
        for word, count in chunk_count.items():
            words_counts[word] += count

    print(f"Parallelizing Pre Tokenization cost: {time.time() - parallel_pre_tokenziation_start}")

    #  -------------------------
    # | Normal Pre Tokenization |
    #  -------------------------
    # normal_pre_tokenziation_start = time.time()
    # with open(input_path, "rb") as f:
    #     content = f.read()
    # sentences = re.split(b"|".join(re.escape(token).encode("utf-8") for token in special_tokens), content)

    # words_counts = defaultdict(int)
    # for sentence in sentences:
    #     matches = re.finditer(pre_tokenization_pattern, sentence.decode('utf-8', errors='replace'))
    #     for match in matches:
    #         word_bytes = tuple(match.group().encode("utf-8"))
    #         words_counts[word_bytes] += 1
    # print(f"Normal Pre Tokenization cost: {time.time() - normal_pre_tokenziation_start}")

    merges: list[tuple[bytes, bytes]] = []

    #  -----------------
    # | Optimized Merge |
    #  -----------------
    merge_start = time.time()
    pair_counts = defaultdict(int)
    pair_to_word_bytes = defaultdict(set)
    for word_bytes, count in words_counts.items():
        pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
        for pair in pairs:
            pair_counts[pair] += count
            pair_to_word_bytes[pair].add(word_bytes)
    while next_index < vocab_size:
        max_pair = max(pair_counts, key=lambda x: (pair_counts[x], (vocab[x[0]], vocab[x[1]])))
        index1, index2 = max_pair
        merges.append((vocab[index1], vocab[index2]))
        vocab[next_index] = vocab[index1] + vocab[index2]
        affected_word_bytes = pair_to_word_bytes[max_pair].copy()
        
        for affected in affected_word_bytes:
            count = words_counts[affected]
            new_word_bytes = []
            i = 0
            while i < len(affected):
                if i < len(affected) - 1 and affected[i] == index1 and affected[i + 1] == index2:
                    new_word_bytes.append(next_index)
                    i += 2
                else:
                    new_word_bytes.append(affected[i])
                    i += 1
            new_word = tuple(new_word_bytes)
            words_counts[new_word] += count

            for pair in zip(affected[:-1], affected[1:]):
                pair_counts[pair] -= count
                pair_to_word_bytes[pair].discard(affected)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    del pair_to_word_bytes[pair]

            new_pairs = list(zip(new_word[:-1], new_word[1:]))
            for pair in new_pairs:
                pair_counts[pair] += count
                pair_to_word_bytes[pair].add(new_word)

        next_index += 1
    print(f"Merge cost: {time.time() - merge_start}")

    #  --------------
    # | Normal Merge |
    #  --------------
    # merge_start = time.time()
    # while next_index < vocab_size:
    #     counts = defaultdict(int)
    #     for word_bytes, count in words_counts.items():
    #         for i in range(len(word_bytes) - 1):
    #             pair = (word_bytes[i], word_bytes[i + 1])
    #             counts[pair] += count
    #     pair = max(counts, key=lambda x: (counts[x], (vocab[x[0]], vocab[x[1]])))
    #     index1, index2 = pair

    #     merges.append((vocab[index1], vocab[index2]))
    #     vocab[next_index] = vocab[index1] + vocab[index2]

    #     new_words_counts = {}
    #     for word_bytes, count in words_counts.items():
    #         new_word_bytes = []
    #         i = 0
    #         while i < len(word_bytes):
    #             if i < len(word_bytes) - 1 and word_bytes[i] == index1 and word_bytes[i + 1] == index2:
    #                 new_word_bytes.append(next_index)
    #                 i += 2
    #             else:
    #                 new_word_bytes.append(word_bytes[i])
    #                 i += 1
    #         new_words_counts[tuple(new_word_bytes)] = count
    #     words_counts = new_words_counts
    #     next_index += 1
    # print(f"Merge cost: {time.time() - merge_start}")
    return vocab, merges

def process_chunk(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict:
    words_counts = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    sentences = re.split(b"|".join(re.escape(token).encode("utf-8") for token in special_tokens), chunk)

    words_counts = defaultdict(int)
    for sentence in sentences:
        matches = re.finditer(pre_tokenization_pattern, sentence.decode('utf-8', errors='replace'))
        for match in matches:
            word_bytes = tuple(match.group().encode("utf-8"))
            words_counts[word_bytes] += 1

    return words_counts
