import httpx
from collections import Counter

BASE_URL = "https://api.spotify.com/v1"
TIMEOUT = 30


class SpotifyClient:
    def __init__(self, access_token: str):
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict = None) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{BASE_URL}{path}", headers=self.headers, params=params, timeout=TIMEOUT)
            if r.status_code == 204:
                return {}
            r.raise_for_status()
            return r.json()

    async def _put(self, path: str, json: dict = None) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.put(f"{BASE_URL}{path}", headers=self.headers, json=json, timeout=TIMEOUT)
            if r.status_code not in (200, 204):
                r.raise_for_status()

    async def _post(self, path: str, json: dict = None) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{BASE_URL}{path}", headers=self.headers, json=json, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json() if r.content else {}

    async def get_me(self) -> dict:
        return await self._get("/me")

    async def get_top_tracks(self, limit=50, time_range="medium_term") -> list:
        data = await self._get("/me/top/tracks", {"limit": limit, "time_range": time_range})
        return data.get("items", [])

    async def get_top_artists(self, limit=20) -> list:
        data = await self._get("/me/top/artists", {"limit": limit})
        return data.get("items", [])

    async def get_recent(self, limit=50) -> list:
        data = await self._get("/me/player/recently-played", {"limit": limit})
        return [item["track"] for item in data.get("items", []) if item.get("track")]

    async def get_discovery_tracks(self, top_artist_ids: list, familiar_ids: set) -> list:
        """Find new tracks via related artists (Spotify deprecated /recommendations for new apps)."""
        discovery = []
        seen_artists = set(top_artist_ids)

        for artist_id in top_artist_ids[:5]:
            try:
                data = await self._get(f"/artists/{artist_id}/related-artists")
                for related in data.get("artists", [])[:4]:
                    if related["id"] in seen_artists:
                        continue
                    seen_artists.add(related["id"])
                    try:
                        tracks_data = await self._get(
                            f"/artists/{related['id']}/top-tracks",
                            {"market": "from_token"},
                        )
                        for track in tracks_data.get("tracks", [])[:3]:
                            if track.get("id") and track["id"] not in familiar_ids:
                                discovery.append(track)
                        if len(discovery) >= 50:
                            return discovery
                    except Exception:
                        continue
            except Exception:
                continue

        return discovery

    async def get_player_state(self) -> dict:
        try:
            return await self._get("/me/player")
        except Exception:
            return {}

    async def play_tracks(self, track_uris: list) -> None:
        await self._put("/me/player/play", {"uris": track_uris[:50]})

    async def pause(self) -> None:
        await self._put("/me/player/pause")

    async def resume(self) -> None:
        await self._put("/me/player/play")

    async def next_track(self) -> None:
        async with httpx.AsyncClient() as client:
            await client.post(f"{BASE_URL}/me/player/next", headers=self.headers, timeout=TIMEOUT)

    async def prev_track(self) -> None:
        async with httpx.AsyncClient() as client:
            await client.post(f"{BASE_URL}/me/player/previous", headers=self.headers, timeout=TIMEOUT)

    async def build_candidate_tracks(self) -> tuple[list, set]:
        """Returns (enriched_track_list, familiar_ids)"""
        top_long, top_medium, recent, top_artists = await _gather(
            self.get_top_tracks(50, "long_term"),
            self.get_top_tracks(50, "medium_term"),
            self.get_recent(50),
            self.get_top_artists(20),
        )

        # Build artist -> genres map from top artists
        artist_genres: dict = {a["id"]: a.get("genres", []) for a in top_artists}

        familiar_ids: set = set()
        familiar_tracks: dict = {}
        for t in top_long + top_medium + recent:
            if t and t.get("id") and t["id"] not in familiar_ids:
                familiar_ids.add(t["id"])
                familiar_tracks[t["id"]] = t

        top_artist_ids = [a["id"] for a in top_artists[:5]]
        discovery_raw = await self.get_discovery_tracks(top_artist_ids, familiar_ids)
        discovery_tracks: dict = {}
        for t in discovery_raw:
            if t and t.get("id") and t["id"] not in familiar_ids:
                discovery_tracks[t["id"]] = t

        candidates = []
        for track_id, track in {**familiar_tracks, **discovery_tracks}.items():
            artists = track.get("artists", [])
            artist_names = [a["name"] for a in artists[:2]]
            first_artist_id = artists[0]["id"] if artists else None
            genres = artist_genres.get(first_artist_id, [])[:3]

            candidates.append({
                "id": track_id,
                "name": track["name"],
                "artist": ", ".join(artist_names),
                "uri": track["uri"],
                "album_art": _album_art(track),
                "familiar": track_id in familiar_ids,
                "genres": genres,
                "popularity": track.get("popularity", 50),
                "duration_ms": track.get("duration_ms", 0),
            })

        return candidates, familiar_ids

    async def get_taste_profile(self, candidates: list) -> dict:
        top_artists_data = await self.get_top_artists(10)
        genre_counts = Counter(g for a in top_artists_data for g in a.get("genres", []))

        return {
            "top_artists": [a["name"] for a in top_artists_data[:5]],
            "genres": [g for g, _ in genre_counts.most_common(5)],
        }


async def _gather(*coros):
    import asyncio
    return await asyncio.gather(*coros)


def _album_art(track: dict) -> str:
    images = track.get("album", {}).get("images", [])
    return images[0]["url"] if images else ""
