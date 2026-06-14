import anthropic
import json

SYSTEM_PROMPT = """You are Algorhythm, an elite AI DJ with encyclopedic music knowledge.
You craft DJ sets that feel like a real DJ performed them — intentional energy arcs,
smooth genre and mood transitions, and discoveries the listener will love.

Golden rules:
1. ENERGY ARC: Start with moderate energy, build toward a peak in the middle, wind down at the end
2. GENRE FLOW: Transition between compatible genres smoothly — don't jump from trap to acoustic folk
3. MOOD: Keep emotional tone consistent or shift it gradually to tell a story
4. DISCOVERY: 20-30% of the set should be [DISCOVERY ★] tracks — this is where you shine
5. NARRATIVE: The set should feel like a journey that matches the requested vibe

Use your deep knowledge of artists and songs to make great sequencing decisions.

Return ONLY valid JSON — no markdown, no code blocks, no explanation outside the JSON."""


def _fmt_tracks(tracks: list) -> str:
    lines = []
    for t in tracks:
        tag = "[familiar]   " if t["familiar"] else "[DISCOVERY ★]"
        genres = ", ".join(t.get("genres", [])[:2]) or "unknown"
        pop = t.get("popularity", 50)
        lines.append(
            f"{tag} {t['artist']} — {t['name']} | "
            f"Genres: {genres} | Popularity: {pop} | ID:{t['id']}"
        )
    return "\n".join(lines)


async def generate_set(vibe: str, taste: dict, candidates: list) -> dict:
    client = anthropic.Anthropic()

    user_msg = f"""USER VIBE: "{vibe}"

TASTE PROFILE:
- Top artists: {', '.join(taste.get('top_artists', [])[:5])}
- Top genres: {', '.join(taste.get('genres', [])[:5])}

CANDIDATE TRACKS ({len(candidates)} total):
{_fmt_tracks(candidates)}

Select 15–20 tracks for the perfect set. Include at least 3 [DISCOVERY ★] tracks.
Order them for smooth DJ flow: compatible genre/mood transitions, intentional energy arc.
Use your knowledge of these artists and songs to make great sequencing decisions.

Respond with this exact JSON (no markdown wrapper):
{{
  "set_name": "Creative, evocative set name",
  "dj_intro": "2 sentences hyping this set like a real DJ. Personal, excited.",
  "track_ids": ["id1", "id2", ...],
  "notes": {{
    "id1": "Short DJ note — why this track, why now, what's the vibe",
    "id2": "..."
  }}
}}"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)
