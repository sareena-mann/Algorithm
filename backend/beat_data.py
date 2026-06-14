"""
Beat data pipeline. Priority order per track:
  1. Spotify /audio-analysis  -> real waveform + beat grid + BPM + key
  2. GetSongBPM API           -> BPM + key + synthetic waveform
  3. Synthetic fallback       -> synthetic waveform only

Output format (stored in each track dict and returned to frontend):
{
  "waveform":  [float, ...]   # 300 normalized loudness points 0-1
  "beats":     [float, ...]   # beat timestamps in seconds
  "sections":  [{start_ratio, dur_ratio, loudness}, ...]
  "bpm":       float | None
  "camelot":   str            # Camelot key e.g. "8A"
  "key_name":  str            # Plain name e.g. "A Minor"
  "wf_source": str            # "spotify" | "getsongbpm" | "synthetic"
}
"""

import os
import math
import random
import httpx
from mixer import get_camelot, KEY_TO_CAMELOT

GETSONGBPM_KEY = os.getenv("GETSONGBPM_API_KEY", "")
WF_POINTS = 300

_cache: dict = {}


async def enrich_track(sp_client, track: dict) -> dict:
    tid = track["id"]
    if tid in _cache:
        return {**track, **_cache[tid]}

    beat = await _from_spotify(sp_client, tid, track.get("duration_ms", 0))
    if not beat:
        beat = await _from_getsongbpm(track.get("artist", ""), track.get("name", ""),
                                      track.get("duration_ms", 0))
    if not beat:
        beat = _synthetic(120, track.get("duration_ms", 0), source="synthetic")

    _cache[tid] = beat
    return {**track, **beat}


async def enrich_tracks(sp_client, tracks: list) -> list:
    import asyncio
    sem = asyncio.Semaphore(4)

    async def _one(t):
        async with sem:
            return await enrich_track(sp_client, t)

    return list(await asyncio.gather(*[_one(t) for t in tracks]))


# ── Sources ───────────────────────────────────────────────────────────────────

async def _from_spotify(sp_client, track_id: str, duration_ms: int) -> dict | None:
    try:
        analysis = await sp_client._get(f"/audio-analysis/{track_id}")
        if not analysis or not analysis.get("segments"):
            return None
        return _process_spotify(analysis, duration_ms)
    except Exception:
        return None


def _process_spotify(analysis: dict, duration_ms: int) -> dict:
    segments  = analysis.get("segments", [])
    beats     = analysis.get("beats", [])
    sections  = analysis.get("sections", [])
    track_inf = analysis.get("track", {})
    dur_s     = max(duration_ms / 1000, 1)

    # Downsample segment loudness to WF_POINTS bars
    raw = [max(0.0, min(1.0, (s.get("loudness_max", -60) + 60) / 60)) for s in segments]
    waveform = _downsample(raw, WF_POINTS)

    beat_times = [b["start"] for b in beats if b.get("confidence", 0) > 0.2]

    sec_data = [
        {
            "s": round(s.get("start", 0) / dur_s, 4),
            "d": round(s.get("duration", 0) / dur_s, 4),
            "l": round(max(0, min(1, (s.get("loudness", -20) + 40) / 40)), 3),
        }
        for s in sections
    ]

    key  = track_inf.get("key", -1)
    mode = track_inf.get("mode", 1)
    bpm  = round(track_inf.get("tempo", 0), 1) or None

    return {
        "waveform":  waveform,
        "beats":     beat_times,
        "sections":  sec_data,
        "bpm":       bpm,
        "camelot":   get_camelot(key, mode) if key != -1 else "",
        "key_name":  "",
        "wf_source": "spotify",
    }


async def _from_getsongbpm(artist: str, title: str, duration_ms: int) -> dict | None:
    if not GETSONGBPM_KEY:
        return None
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.getsongbpm.com/search/",
                params={
                    "api_key": GETSONGBPM_KEY,
                    "type": "both",
                    "lookup": f"song:{title} artist:{artist}",
                },
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()

        song = data.get("song") or (data.get("songs") or [None])[0]
        if not song:
            return None

        bpm      = int(float(song.get("tempo") or 0)) or None
        key_name = song.get("key_of", "")
        camelot  = KEY_TO_CAMELOT.get(key_name, "")

        return _synthetic(bpm or 120, duration_ms, bpm=bpm, camelot=camelot,
                          key_name=key_name, source="getsongbpm")
    except Exception:
        return None


# ── Synthetic waveform ────────────────────────────────────────────────────────

def _synthetic(bpm: float, duration_ms: int, *, bpm_out=None, camelot="",
               key_name="", source="synthetic") -> dict:
    dur_s  = max(duration_ms / 1000, 60)
    random.seed(int(bpm * 100 + duration_ms))  # deterministic per track

    waveform = []
    for i in range(WF_POINTS):
        pos = i / WF_POINTS
        # Typical energy arc: intro → build → verse → chorus → verse → outro
        if   pos < 0.05:  base = 0.20
        elif pos < 0.15:  base = 0.20 + (pos - 0.05) / 0.10 * 0.35
        elif pos < 0.35:  base = 0.55
        elif pos < 0.45:  base = 0.55 + (pos - 0.35) / 0.10 * 0.25
        elif pos < 0.60:  base = 0.80
        elif pos < 0.70:  base = 0.55
        elif pos < 0.80:  base = 0.75
        elif pos < 0.92:  base = 0.75 - (pos - 0.80) / 0.12 * 0.50
        else:             base = 0.22

        beat_phase = (pos * dur_s * bpm / 60) % 1.0
        beat_boost = max(0, 1 - beat_phase * 8) * 0.18
        noise      = (random.random() - 0.5) * 0.08
        waveform.append(round(min(1.0, max(0.04, base + beat_boost + noise)), 3))

    beats = []
    t = 0.0
    interval = 60.0 / bpm
    while t < dur_s:
        beats.append(round(t, 3))
        t += interval

    return {
        "waveform":  waveform,
        "beats":     beats,
        "sections":  [],
        "bpm":       bpm_out if bpm_out is not None else (bpm if source != "synthetic" else None),
        "camelot":   camelot,
        "key_name":  key_name,
        "wf_source": source,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _downsample(data: list, n: int) -> list:
    if not data:
        return []
    if len(data) <= n:
        return [round(v, 3) for v in data]
    step = len(data) / n
    return [round(max(data[int(i * step): int((i + 1) * step)] or [0]), 3) for i in range(n)]
