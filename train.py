import argparse
import numpy as np
from cs336_basics.transformer import Transformer
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.cosine_schedule import cosine_schedule


def load_tokenized_dataset(path):
    arr = np.load(path, mmap_mode="r")
    return arr


def train_loop(
    dataset_path,
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    batch_size,
    total_iters,
    lr_max,
    lr_min,
    warmup,
    cosine_iters_per_cycle,
    grad_clip,
    device,
    ckpt_path,
):
    dataset = load_tokenized_dataset(dataset_path)
    print(f"Dataset loaded: {dataset.shape}")

    model = Transformer(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        device,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr_max)
    iteration = 0

    try:
        iteration = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Loaded checkpoint at iter {iteration}")
    except:
        print(f"No checkpoint loaded, starting from scratch")

    for it in range(iteration, total_iters):
        x, y = data_loading(dataset, batch_size, context_length, device)

        logits = model(x)
        logits_flat = logits[:, :-1].reshape(-1, vocab_size)
        targets_flat = y[:, :-1].reshape(-1)

        loss = cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), grad_clip)

        lr = cosine_schedule(it, lr_max, lr_min, warmup, cosine_iters_per_cycle)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()

        if it % 50 == 0:
            print(f"Iter {it} loss {loss.item():.4f} lr {lr:.6f}")
        if it % 500 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)
            print("Checkpoint saved")

    save_checkpoint(model, optimizer, total_iters, ckpt_path)
    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--cosine_iters", type=int, default=20000)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")

    args = parser.parse_args()

    train_loop(
        dataset_path=args.dataset,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        total_iters=args.iters,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup=args.warmup,
        cosine_iters_per_cycle=args.cosine_iters,
        grad_clip=args.grad_clip,
        device=args.device,
        ckpt_path=args.ckpt,
    )
