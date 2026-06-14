"""
Synthetic transition generator — Option C from the architecture plan.

Creates mathematically perfect triplets (context_a, bridge, context_b)
from pure synthesis: no real audio files needed. The bridge is constructed
so the model learns structure and timing without ever seeing human-made mixes.

Synthesis approach:
  - Each "track" is additive synthesis: drum hits (sine bursts), bass (sawtooth),
    pad (filtered noise), melody (sine waves at key-correct pitches)
  - Transition is a crossfade that swaps drum timbre + key/BPM simultaneously
  - BPM, key, and genre are all hardcoded so the conditioning vector is exact
"""

import math
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from .constants import (
    SAMPLE_RATE, CONTEXT_A_SECS, CONTEXT_B_SECS,
    BRIDGE_SECS_MIN, BRIDGE_SECS_MAX, GENRES, build_conditioning,
)

BPM_OPTIONS   = [90, 100, 110, 120, 124, 126, 128, 130, 140, 150, 174]
CAMELOT_KEYS  = [f"{n}{m}" for n in range(1, 13) for m in "AB"]

MIDI_NOTE = {  # Camelot → MIDI root note (C4=60)
    "1A": 57,  "1B": 64,   # A min / C maj
    "2A": 64,  "2B": 71,   # E min / G maj
    "3A": 59,  "3B": 66,   # B min / D maj
    "4A": 54,  "4B": 61,   # F# min / A maj
    "5A": 61,  "5B": 68,   # C# min / E maj
    "6A": 56,  "6B": 63,   # G# min / B maj
    "7A": 63,  "7B": 70,   # Eb min / F# maj
    "8A": 57,  "8B": 60,   # A min / C maj (octave alias for demo)
    "9A": 62,  "9B": 65,   # D min / F maj
    "10A": 67, "10B": 70,  # G min / Bb maj
    "11A": 62, "11B": 65,  # D min / F maj (alias)
    "12A": 57, "12B": 60,  # A min / C maj (alias)
}


class SyntheticTransitionGenerator(Dataset):
    def __init__(self, n_pairs: int = 20000, seed: int = 42):
        self.n     = n_pairs
        self.seed  = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)

        bpm_a    = rng.choice(BPM_OPTIONS)
        bpm_b    = rng.choice(BPM_OPTIONS)
        key_a    = rng.choice(CAMELOT_KEYS)
        key_b    = rng.choice(CAMELOT_KEYS)
        genre_a  = rng.choice(GENRES)
        genre_b  = rng.choice(GENRES)

        bridge_secs = rng.uniform(BRIDGE_SECS_MIN, BRIDGE_SECS_MAX)
        T_a         = int(CONTEXT_A_SECS * SAMPLE_RATE)
        T_b         = int(CONTEXT_B_SECS * SAMPLE_RATE)
        T_bridge    = int(bridge_secs * SAMPLE_RATE)

        ctx_a  = _synth_track(bpm_a, key_a, genre_a, T_a,      rng, energy_end=True)
        bridge = _synth_bridge(bpm_a, bpm_b, key_a, key_b, T_bridge, rng)
        ctx_b  = _synth_track(bpm_b, key_b, genre_b, T_b,      rng, energy_end=False)

        cond = build_conditioning(bpm_a, bpm_b, key_a, key_b, genre_a, genre_b)

        return {
            "context_a":    ctx_a.unsqueeze(0),    # (1, T_a)
            "bridge":       bridge.unsqueeze(0),   # (1, T_bridge)
            "context_b":    ctx_b.unsqueeze(0),    # (1, T_b)
            "conditioning": cond,
            "source":       "synthetic",
        }


# ── Synthesis helpers ──────────────────────────────────────────────────────

def _hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _sine(freq: float, n_samples: int, phase: float = 0.0) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    return torch.sin(2 * math.pi * freq * t + phase)


def _synth_track(bpm: float, key: str, genre: str, n_samples: int,
                 rng: random.Random, energy_end: bool) -> torch.Tensor:
    """Synthesize a fragment that sounds like a loop of the given track params."""
    audio = torch.zeros(n_samples)
    root_hz = _hz(MIDI_NOTE.get(key, 60))

    beat_samples = int(60 / bpm * SAMPLE_RATE)
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE

    # Kick: sine burst at each beat
    for beat in range(n_samples // beat_samples + 1):
        onset = beat * beat_samples
        if onset >= n_samples:
            break
        dur = min(int(0.08 * SAMPLE_RATE), n_samples - onset)
        env = torch.exp(-torch.arange(dur, dtype=torch.float32) / (dur * 0.3))
        kick = env * torch.sin(2 * math.pi * 60 * torch.arange(dur, dtype=torch.float32) / SAMPLE_RATE)
        audio[onset: onset + dur] += 0.3 * kick

    # Bass: sawtooth at root
    saw = 2 * ((root_hz * t) % 1.0) - 1
    bass_env = torch.ones(n_samples)
    if not energy_end:
        bass_env[:n_samples // 4] = torch.linspace(0, 1, n_samples // 4)
    audio += 0.15 * saw * bass_env

    # Pad: filtered noise around root freq
    noise = torch.randn(n_samples) * 0.05
    pad   = noise * (0.5 + 0.5 * _sine(root_hz * 2, n_samples, rng.random()))
    audio += 0.1 * pad

    # Energy fade at end if this is leading into a transition
    if energy_end:
        fade_start = int(n_samples * 0.8)
        fade = torch.ones(n_samples)
        fade[fade_start:] = torch.linspace(1.0, 0.7, n_samples - fade_start)
        audio *= fade

    return audio.clamp(-1, 1)


def _synth_bridge(bpm_a: float, bpm_b: float, key_a: str, key_b: str,
                  n_samples: int, rng: random.Random) -> torch.Tensor:
    """Crossfade bridge: track A fades out while track B fades in."""
    seg_a = _synth_track(bpm_a, key_a, "techno", n_samples, rng, energy_end=True)
    seg_b = _synth_track(bpm_b, key_b, "techno", n_samples, rng, energy_end=False)

    fade_out = torch.linspace(1.0, 0.0, n_samples) ** 1.5
    fade_in  = torch.linspace(0.0, 1.0, n_samples) ** 1.5

    bridge = seg_a * fade_out + seg_b * fade_in
    return bridge.clamp(-1, 1)
