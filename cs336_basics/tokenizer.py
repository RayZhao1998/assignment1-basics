from typing import Iterable
import regex as re

pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    def from_files(cls, vocab_filepath, merge_filepath, special_tokens=None):
        return NotImplemented

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

    def decode(self, ids: list[int]) -> str:
        list = []
        for id in ids:
            list.append(self.vocab[id])
        return b"".join(list).decode("utf-8", errors="replace")
