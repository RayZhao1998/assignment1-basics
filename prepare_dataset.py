import argparse
import numpy as np

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import BPETokenizer


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--output_tokens", type=str, default="tokens.npy")
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()

    text = read_text(args.input)
    print("Loaded text")

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special,
    )
    print("Trained BPE")

    tokenizer = BPETokenizer(vocab, merges, special_tokens=args.special)
    ids = tokenizer.encode_parallel(text, num_workers=14)
    print("Tokenized")

    arr = np.array(ids, dtype=np.uint16)
    np.save(args.output_tokens, arr)
    print(f"Saved to {args.output_tokens}")


if __name__ == "__main__":
    main()
