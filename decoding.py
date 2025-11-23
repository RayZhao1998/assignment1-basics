import torch
import argparse
from cs336_basics.tokenizer import BPETokenizer


def load_tokenizer():
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="data/tinystories_valid_tokenizer/tinystories_vocab.json",
        merges_filepath="data/tinystories_valid_tokenizer/tinystories_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    return tokenizer


def load_model(checkpoint_path: str, device: str):
    from cs336_basics.transformer import Transformer

    ckpt = torch.load(checkpoint_path)
    config = ckpt["model_state"]

    model = Transformer(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device=device,
    )

    model.load_state_dict(config)
    model.to(device)
    return model


def sample_next_token(logits, temperature=1.0, top_p=1.0):
    from cs336_basics.softmax import softmax

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    probs = softmax(logits, -1)

    if top_p is None or top_p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative <= top_p
    if not torch.any(mask):
        mask[0] = True

    cutoff = torch.nonzero(mask)[-1].item()
    mask[: cutoff + 1] = True

    truncated_probs = sorted_probs * mask
    truncated_probs /= truncated_probs.sum()

    sampled = torch.multinomial(truncated_probs, 1)
    next_id = sorted_idx[sampled]

    return int(next_id.item())


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt_ids: torch.Tensor,
    max_tokens: int,
    device,
    temperature=1.0,
    top_p=1.0,
):
    model.eval()
    ids = prompt_ids.to(device)

    eos_id = tokenizer.vocab_reverse[b"<|endoftext|>"]

    for _ in range(max_tokens):
        logits = model(ids.unsqueeze(0))
        last_logits = logits[0, -1]

        next_id = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        next_id_tensor = torch.tensor([next_id], dtype=torch.long, device=device)

        ids = torch.cat([ids, next_id_tensor], dim=0)

        if next_id == eos_id:
            break
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()
    device = args.device

    tokenizer = load_tokenizer()

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

    model = load_model(args.checkpoint, device)
    full_ids = decode(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    text = tokenizer.decode(full_ids.tolist())
    print(text)
