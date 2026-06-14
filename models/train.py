"""
Training loop for TransitionDiT.

Usage:
  python train.py                          # synthetic-only
  python train.py --data_dir /path/to/real # blend real + synthetic

Checkpoints saved to: checkpoints/dit_epoch{N}.pt
Resumes automatically if checkpoints/ exists.
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from arch.codec  import AudioCodec
from arch.dit    import TransitionDiT, add_noise, T_STEPS, LATENT_DIM
from data.dataset import BlendedTransitionDataset, SAMPLE_RATE

DEVICE       = "mps" if torch.backends.mps.is_available() else \
               "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 8
LR           = 1e-4
N_EPOCHS     = 100
SAVE_EVERY   = 5
CKPT_DIR     = Path("checkpoints")


def collate(batch: list[dict]) -> dict:
    """Pad bridge tensors to the same length within a batch."""
    max_bridge = max(b["bridge"].shape[-1] for b in batch)
    out = {}
    for key in ("context_a", "bridge", "context_b", "conditioning"):
        tensors = [b[key] for b in batch]
        if key == "bridge":
            tensors = [F.pad(t, (0, max_bridge - t.shape[-1])) for t in tensors]
        out[key] = torch.stack(tensors)
    return out


def encode_batch(codec: AudioCodec, batch: dict, device: str) -> dict:
    """Encode raw audio to latent tokens. Called per batch inside training loop."""
    ctx_a = codec.encode(batch["context_a"].to(device))    # (B, T_a, D)
    bridge = codec.encode(batch["bridge"].to(device))       # (B, T_br, D)
    ctx_b  = codec.encode(batch["context_b"].to(device))   # (B, T_b, D)
    cond   = batch["conditioning"].to(device)               # (B, COND_DIM)
    return {"context_a": ctx_a, "bridge": bridge, "context_b": ctx_b, "cond": cond}


def train_step(model, batch_latents, optimizer) -> float:
    ctx_a  = batch_latents["context_a"]   # (B, T_a, D)
    bridge = batch_latents["bridge"]      # (B, T_br, D)
    ctx_b  = batch_latents["context_b"]  # (B, T_b, D)
    cond   = batch_latents["cond"]        # (B, COND_DIM)

    B = bridge.shape[0]
    t = torch.randint(0, T_STEPS, (B,), device=bridge.device)

    noisy_bridge, epsilon = add_noise(bridge, t)
    eps_pred = model(ctx_a, noisy_bridge, ctx_b, t, cond)

    loss = F.mse_loss(eps_pred, epsilon)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default=None,
                        help="Path to real pairs directory (optional)")
    parser.add_argument("--synth_n",   type=int, default=20000)
    parser.add_argument("--real_ratio",type=float, default=0.5)
    parser.add_argument("--batch_size",type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--epochs",    type=int, default=N_EPOCHS)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--n_heads",   type=int, default=8)
    parser.add_argument("--n_layers",  type=int, default=12)
    args = parser.parse_args()

    print(f"Training on: {DEVICE}")
    CKPT_DIR.mkdir(exist_ok=True)

    print("Loading audio codec...")
    codec = AudioCodec(device=DEVICE)

    print("Building dataset...")
    dataset = BlendedTransitionDataset(
        data_dir=args.data_dir,
        synth_n=args.synth_n,
        real_ratio=args.real_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    model = TransitionDiT(model_dim=args.model_dim, n_heads=args.n_heads, n_layers=args.n_layers).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from latest checkpoint if it exists
    start_epoch = 1
    ckpts = sorted(CKPT_DIR.glob("dit_epoch*.pt"))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from {latest}")
        state = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps    = 0

        for batch in loader:
            with torch.no_grad():
                latents = encode_batch(codec, batch, DEVICE)
            loss = train_step(model, latents, optimizer)
            total_loss += loss
            n_steps    += 1

            if n_steps % 50 == 0:
                print(f"  epoch {epoch}  step {n_steps}  loss {loss:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % SAVE_EVERY == 0:
            path = CKPT_DIR / f"dit_epoch{epoch:04d}.pt"
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)
            print(f"  Saved checkpoint: {path}")


if __name__ == "__main__":
    main()
