import torch

SAMPLE_RATE     = 44100
CONTEXT_A_SECS  = 15
CONTEXT_B_SECS  = 10
BRIDGE_SECS_MIN = 4
BRIDGE_SECS_MAX = 10

GENRES = [
    "house", "techno", "trance", "drum_and_bass",
    "hip_hop", "r_and_b", "pop", "ambient",
    "disco", "funk", "jazz", "indie",
]
N_GENRES = len(GENRES)
COND_DIM = 4 + 2 * N_GENRES  # 28


def _camelot_compat(k1: str, k2: str) -> float:
    if not k1 or not k2:
        return 0.5
    if k1 == k2:
        return 1.0
    num1, mode1 = int(k1[:-1]), k1[-1]
    num2, mode2 = int(k2[:-1]), k2[-1]
    if mode1 == mode2:
        return 0.75 if abs(num1 - num2) in (1, 11) else 0.25
    if num1 == num2:
        return 0.75
    return 0.25


def build_conditioning(
    bpm_a: float, bpm_b: float,
    key_a: str,   key_b: str,
    genre_a: str, genre_b: str,
) -> torch.Tensor:
    scalars = torch.tensor([
        bpm_a / 200.0,
        bpm_b / 200.0,
        min(bpm_a, bpm_b) / max(bpm_a, bpm_b + 1e-6),
        _camelot_compat(key_a, key_b),
    ], dtype=torch.float32)

    ga = torch.zeros(N_GENRES)
    gb = torch.zeros(N_GENRES)
    if genre_a in GENRES:
        ga[GENRES.index(genre_a)] = 1.0
    if genre_b in GENRES:
        gb[GENRES.index(genre_b)] = 1.0

    return torch.cat([scalars, ga, gb])  # (COND_DIM,)
