from typing import Iterable
import regex as re

pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _bpe_worker(args):
    tokenizer, chunk = args
    return tokenizer.encode(chunk)

def _parse_merge_line(line: str):
    assert line.startswith("['") and line.endswith("']")

    inner = line[2:-2]

    split_index = inner.find("', '")
    left = inner[:split_index]
    right = inner[split_index + 4:]

    return left, right

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None) -> None:
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ids = []
        for pair in merges:
            token1_id = self.vocab_reverse[pair[0]]
            token2_id = self.vocab_reverse[pair[1]]
            merge_id = self.vocab_reverse[pair[0] + pair[1]]
            self.merge_ids.append((token1_id, token2_id, merge_id))
        self.byte_to_token = {}
        for token_id, token_bytes in vocab.items():
            if len(token_bytes) == 1:
                byte_val = token_bytes[0]
                self.byte_to_token[byte_val] = token_id
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        from tests.common import gpt2_bytes_to_unicode

        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {}
        for unicode_token, token_id in raw_vocab.items():
            token_bytes = bytes([byte_decoder[c] for c in unicode_token])
            vocab[int(token_id)] = token_bytes

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                left_unicode, right_unicode = _parse_merge_line(line)
                left_bytes = bytes([byte_decoder[c] for c in left_unicode])
                right_bytes = bytes([byte_decoder[c] for c in right_unicode])

                merges.append((left_bytes, right_bytes))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        splitted_text = []
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            splitted_text = re.split("(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")", text)
        else:
            splitted_text = [text]

        result: list[int] = []
        for subtext in splitted_text:
            if not subtext:
                continue
            if self.special_tokens and subtext in self.special_tokens:
                result.append(self.vocab_reverse[subtext.encode("utf-8")])
            else:
                words: list[str] = re.findall(pre_tokenization_pattern, subtext)
                for word in words:
                    word_bytes = list(word.encode("utf-8"))
                    word_tokens = [self.byte_to_token[b] for b in word_bytes]
                    for token1_id, token2_id, merge_id in self.merge_ids:
                        new_word_tokens = []
                        i = 0
                        while i < len(word_tokens):
                            if i + 1 < len(word_tokens) and word_tokens[i] == token1_id and word_tokens[i + 1] == token2_id:
                                new_word_tokens.append(merge_id)
                                i += 2
                            else:
                                new_word_tokens.append(word_tokens[i])
                                i += 1
                        word_tokens = new_word_tokens
                    result.extend(word_tokens)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def encode_parallel(self, text: str, num_workers: int = 4) -> list[int]:
        import multiprocessing as mp

        length = len(text)
        if length == 0:
            return []

        chunk_size = max(1, length // num_workers)
        chunks = [text[i:i + chunk_size] for i in range(0, length, chunk_size)]

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_bpe_worker, [(self, c) for c in chunks])

        merged: list[int] = []
        for r in results:
            merged.extend(r)
        return merged


    def decode(self, ids: list[int]) -> str:
        list = []
        for id in ids:
            list.append(self.vocab[id])
        return b"".join(list).decode("utf-8", errors="replace")
