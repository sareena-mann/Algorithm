"""
Inference — given two audio files, generate a transition bridge between them.

Usage:
  python infer.py track_a.wav track_b.wav \
    --bpm_a 124 --bpm_b 128 \
    --key_a 8A  --key_b 9A  \
    --genre_a house --genre_b techno \
    --bridge_secs 8 \
    --checkpoint checkpoints/dit_epoch0100.pt \
    --out transition.wav

Output: a WAV file containing:
  [last 15s of A] + [generated bridge] + [first 10s of B]
"""

import argparse
import torch
import torchaudio
from pathlib import Path

from arch.codec   import AudioCodec, SAMPLE_RATE
from arch.dit     import TransitionDiT, ddpm_sample, LATENT_DIM
from data.dataset import build_conditioning, CONTEXT_A_SECS, CONTEXT_B_SECS

DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path: str) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("track_a",       type=str)
    parser.add_argument("track_b",       type=str)
    parser.add_argument("--bpm_a",       type=float, default=120)
    parser.add_argument("--bpm_b",       type=float, default=120)
    parser.add_argument("--key_a",       type=str,   default="")
    parser.add_argument("--key_b",       type=str,   default="")
    parser.add_argument("--genre_a",     type=str,   default="house")
    parser.add_argument("--genre_b",     type=str,   default="house")
    parser.add_argument("--bridge_secs", type=float, default=8.0)
    parser.add_argument("--checkpoint",  type=str,   required=True)
    parser.add_argument("--out",         type=str,   default="transition.wav")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    print("Loading codec...")
    codec = AudioCodec(device=DEVICE)

    print("Loading model...")
    model = TransitionDiT(model_dim=512, n_heads=8, n_layers=12).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    # Load audio
    a_audio = load_audio(args.track_a)
    b_audio = load_audio(args.track_b)

    T_a = int(CONTEXT_A_SECS * SAMPLE_RATE)
    T_b = int(CONTEXT_B_SECS * SAMPLE_RATE)
    T_bridge = int(args.bridge_secs * SAMPLE_RATE)

    context_a = a_audio[:, -T_a:].unsqueeze(0)  # (1, 1, T_a)
    context_b = b_audio[:, :T_b ].unsqueeze(0)  # (1, 1, T_b)

    # Encode to latents
    print("Encoding audio to latent space...")
    with torch.no_grad():
        lat_a = codec.encode(context_a)  # (1, T_a_tok, D)
        lat_b = codec.encode(context_b)  # (1, T_b_tok, D)

    bridge_tokens = codec.secs_to_tokens(args.bridge_secs)
    cond = build_conditioning(
        args.bpm_a, args.bpm_b,
        args.key_a, args.key_b,
        args.genre_a, args.genre_b,
    ).unsqueeze(0).to(DEVICE)  # (1, COND_DIM)

    # Generate bridge latents via DDPM sampling
    print(f"Generating {args.bridge_secs}s bridge ({bridge_tokens} tokens)...")
    lat_bridge = ddpm_sample(model, lat_a, lat_b, cond, bridge_tokens, device=DEVICE)

    # Decode all three parts
    print("Decoding...")
    with torch.no_grad():
        audio_a      = codec.decode(lat_a)      # (1, 1, T_a_audio)
        audio_bridge = codec.decode(lat_bridge) # (1, 1, T_bridge_audio)
        audio_b      = codec.decode(lat_b)      # (1, 1, T_b_audio)

    # Crossfade the seams (4096 samples ~= 93ms)
    fade_n = 4096
    fade_out = torch.linspace(1, 0, fade_n, device=DEVICE)
    fade_in  = torch.linspace(0, 1, fade_n, device=DEVICE)

    audio_a[..., -fade_n:] *= fade_out
    audio_bridge[..., :fade_n]  *= fade_in
    audio_bridge[..., -fade_n:] *= fade_out
    audio_b[..., :fade_n]  *= fade_in

    out = torch.cat([audio_a, audio_bridge, audio_b], dim=-1).squeeze(0).cpu()
    torchaudio.save(args.out, out, SAMPLE_RATE)
    duration = out.shape[-1] / SAMPLE_RATE
    print(f"Saved {args.out} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
