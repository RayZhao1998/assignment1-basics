import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

import pathlib
import pytest
data_folder = (pathlib.Path(__file__).resolve().parent.parent) / "data"
TINYSTORIES_VALID_DIR = data_folder / "tinystories_valid_tokenizer"
TINYSTORIES_ARTIFACTS_DIR = data_folder / "tinystories_tokenizer"


def _save_tokenizer_artifacts(vocab, merges, output_dir):
    """Persist the trained tokenizer so we can inspect it outside the test run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"

    # Convert vocab to GPT-2 format: {token_unicode: index}
    # First, convert bytes to unicode using the same mapping as GPT-2
    from .common import gpt2_bytes_to_unicode
    byte_to_unicode = gpt2_bytes_to_unicode()

    gpt2_vocab = {}
    for idx, token_bytes in vocab.items():
        # Convert each byte to its unicode representation
        token_unicode = ''.join([byte_to_unicode[byte] for byte in token_bytes])
        gpt2_vocab[token_unicode] = idx

    # Save vocab as JSON
    with open(vocab_path, "w", encoding="utf-8") as vocab_file:
        json.dump(gpt2_vocab, vocab_file, ensure_ascii=False, indent=2)

    # Save merges as text file in HuggingFace format
    with open(merges_path, "w", encoding="utf-8") as merges_file:
        for left, right in merges:
            # Convert bytes to unicode representation
            left_unicode = ''.join([byte_to_unicode[byte] for byte in left])
            right_unicode = ''.join([byte_to_unicode[byte] for byte in right])
            merges_file.write(f"['{left_unicode}', '{right_unicode}']\n")

@pytest.mark.skip()
def test_train_bpe_on_tiny_story_valid():
    start_time = time.time()
    input_path = data_folder / "TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"])
    _save_tokenizer_artifacts(vocab, merges, TINYSTORIES_VALID_DIR)
    end_time = time.time()

    assert(end_time - start_time <= 120)

@pytest.mark.skip()
def test_train_bpe_on_tiny_story_train():
    start_time = time.time()
    input_path = data_folder / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"])
    _save_tokenizer_artifacts(vocab, merges, TINYSTORIES_ARTIFACTS_DIR)
    end_time = time.time()

    assert(end_time - start_time <= 120)

    # Compare with HuggingFace tokenizer results
    reference_vocab_path = (pathlib.Path(__file__).resolve().parent.parent) / "hf_tokenizer" / "TinyStory" / "vocab.json"
    reference_merges_path = (pathlib.Path(__file__).resolve().parent.parent) / "hf_tokenizer" / "TinyStory" / "merges.txt"

    # Compare the learned merges to the expected output merges
    from .common import gpt2_bytes_to_unicode
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(reference_merges_path) as f:
        hf_merges = []
        for line in f:
            line = line.strip()
            if line:
                # Parse format: ['Ġ', 't']
                import ast
                parsed = ast.literal_eval(line)
                hf_merges.append((
                    bytes([gpt2_byte_decoder[token] for token in parsed[0]]),
                    bytes([gpt2_byte_decoder[token] for token in parsed[1]])
                ))

    # Compare merges (should be identical)
    assert merges == hf_merges, f"Merges don't match. Our merges count: {len(merges)}, HF merges count: {len(hf_merges)}"

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }

    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys), f"Vocab keys don't match. Our keys: {len(set(vocab.keys()))}, Reference keys: {len(set(reference_vocab.keys))}"
    assert set(vocab.values()) == set(reference_vocab.values), f"Vocab values don't match. Our values: {len(set(vocab.values()))}, Reference values: {len(set(reference_vocab.values))}"
