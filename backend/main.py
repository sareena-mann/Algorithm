import os
import time
import secrets
import httpx
from urllib.parse import urlencode

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from spotify import SpotifyClient
from dj import generate_set
from beat_data import enrich_tracks

load_dotenv()

CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")

SCOPES = " ".join([
    "user-read-private",
    "user-read-email",
    "user-top-read",
    "user-read-recently-played",
    "user-library-read",
    "user-read-playback-state",
    "user-modify-playback-state",
    "streaming",
])

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict = {}  # session_id -> {access_token, refresh_token, expires_at, user}


# ─── OAuth helpers ─────────────────────────────────────────────────────────────

async def _exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        })
        r.raise_for_status()
        return r.json()


async def _refresh_token(refresh_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        })
        r.raise_for_status()
        return r.json()


async def require_auth(request: Request) -> dict:
    session_id = request.cookies.get("algorhythm_session")
    if not session_id or session_id not in sessions:
        raise HTTPException(401, "Not authenticated")

    session = sessions[session_id]
    if time.time() >= session.get("expires_at", 0) - 60:
        try:
            data = await _refresh_token(session["refresh_token"])
            session["access_token"] = data["access_token"]
            session["expires_at"] = time.time() + data.get("expires_in", 3600)
            if "refresh_token" in data:
                session["refresh_token"] = data["refresh_token"]
            sessions[session_id] = session
        except Exception:
            raise HTTPException(401, "Session expired. Please reconnect Spotify.")

    return session


# ─── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/login")
async def login(response_obj: Request = None):
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "scope": SCOPES,
        "show_dialog": "false",
    }
    redirect = RedirectResponse(f"{AUTH_URL}?{urlencode(params)}")
    redirect.set_cookie("spotify_state", state, max_age=300, httponly=True)
    return redirect


@app.get("/callback")
async def callback(request: Request, code: str = None, state: str = None, error: str = None):
    if error or not code:
        return RedirectResponse("/?error=access_denied")

    stored_state = request.cookies.get("spotify_state")
    if state != stored_state:
        return RedirectResponse("/?error=state_mismatch")

    try:
        token_data = await _exchange_code(code)
    except Exception:
        return RedirectResponse("/?error=token_exchange_failed")

    sp = SpotifyClient(token_data["access_token"])
    user = await sp.get_me()

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": time.time() + token_data.get("expires_in", 3600),
        "user": {
            "id": user.get("id"),
            "name": user.get("display_name"),
            "email": user.get("email"),
            "image": (user.get("images") or [{}])[0].get("url", ""),
        },
    }

    redirect = RedirectResponse("/")
    redirect.delete_cookie("spotify_state")
    redirect.set_cookie("algorhythm_session", session_id, max_age=86400, httponly=True, samesite="lax")
    return redirect


@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("algorhythm_session")
    if session_id:
        sessions.pop(session_id, None)
    r = RedirectResponse("/")
    r.delete_cookie("algorhythm_session")
    return r


# ─── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/me")
async def get_me(request: Request):
    session = await require_auth(request)
    return session["user"]


class VibeRequest(BaseModel):
    vibe: str


@app.post("/api/generate-set")
async def api_generate_set(body: VibeRequest, request: Request):
    session = await require_auth(request)
    sp = SpotifyClient(session["access_token"])

    candidates, _ = await sp.build_candidate_tracks()
    if not candidates:
        raise HTTPException(400, "Could not load your Spotify library. Make sure you have listening history.")

    taste = await sp.get_taste_profile(candidates)

    try:
        result = await generate_set(body.vibe, taste, candidates)
    except Exception as e:
        raise HTTPException(500, f"DJ brain error: {e}")

    track_map = {t["id"]: t for t in candidates}
    selected = []
    for track_id in result.get("track_ids", []):
        t = track_map.get(track_id)
        if t:
            selected.append({**t, "note": result.get("notes", {}).get(track_id, "")})

    enriched = await enrich_tracks(sp, selected)

    return {
        "set_name": result.get("set_name", "Your Set"),
        "dj_intro": result.get("dj_intro", ""),
        "tracks": enriched,
    }


class PlaySetRequest(BaseModel):
    track_uris: list[str]


@app.post("/api/play-set")
async def api_play_set(body: PlaySetRequest, request: Request):
    session = await require_auth(request)
    sp = SpotifyClient(session["access_token"])

    state = await sp.get_player_state()
    if not state or not state.get("device"):
        raise HTTPException(400, "No active Spotify device. Open Spotify on any device first, then try again.")

    await sp.play_tracks(body.track_uris)
    return {"status": "playing"}


@app.get("/api/player/state")
async def api_player_state(request: Request):
    session = await require_auth(request)
    sp = SpotifyClient(session["access_token"])
    state = await sp.get_player_state()
    if not state:
        return {"playing": False}

    item = state.get("item") or {}
    artists = [a["name"] for a in item.get("artists", [])]
    images = item.get("album", {}).get("images", [])

    return {
        "playing": state.get("is_playing", False),
        "track_name": item.get("name", ""),
        "artist": ", ".join(artists),
        "album_art": images[0]["url"] if images else "",
        "track_id": item.get("id", ""),
        "progress_ms": state.get("progress_ms", 0),
        "duration_ms": item.get("duration_ms", 0),
        "device": state.get("device", {}).get("name", ""),
    }


class ControlRequest(BaseModel):
    action: str  # play | pause | next | prev


@app.post("/api/player/control")
async def api_player_control(body: ControlRequest, request: Request):
    session = await require_auth(request)
    sp = SpotifyClient(session["access_token"])

    if body.action == "pause":
        await sp.pause()
    elif body.action == "play":
        await sp.resume()
    elif body.action == "next":
        await sp.next_track()
    elif body.action == "prev":
        await sp.prev_track()

    return {"status": "ok"}


# ─── Static files ──────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
