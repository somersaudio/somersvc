"""GetSongBPM + Spotify integration for estimating artist vocal keys."""

import math
import os
import requests

from services.job_store import load_config

# Note name to semitone mapping (C=0)
NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "C♯": 1, "Db": 1, "D♭": 1,
    "D": 2, "D#": 3, "D♯": 3, "Eb": 3, "E♭": 3,
    "E": 4, "F": 5, "F#": 6, "F♯": 6, "Gb": 6, "G♭": 6,
    "G": 7, "G#": 8, "G♯": 8, "Ab": 8, "A♭": 8,
    "A": 9, "A#": 10, "A♯": 10, "Bb": 10, "B♭": 10,
    "B": 11,
}

SEMI_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
C4_HZ = 261.63  # Default to octave 4 (mid-range, suits most vocalists)


def _parse_key(key_str: str) -> int | None:
    """Parse a key string like 'C#m', 'Bb', 'F♯' into a semitone (0-11)."""
    clean = key_str.strip().rstrip("mM").strip()
    return NOTE_TO_SEMI.get(clean)


def estimate_artist_key(artist_name: str) -> tuple[str, float]:
    """Estimate an artist's vocal center using Spotify track names + GetSongBPM keys.

    1. Search Spotify for the artist's tracks (search endpoint, no restricted APIs).
    2. Look up each track on GetSongBPM using 'both' search to get musical key.
    3. Circular-average the keys to find the center note.

    Returns (note_name, hz) or ("", 0) on failure.
    """
    config = load_config()
    songbpm_key = config.get(
        "songbpm_api_key", os.environ.get("SOMERSVC_SONGBPM_KEY", "")
    )
    if not songbpm_key:
        return "", 0

    # Get track names from Spotify search (no restricted endpoints needed)
    track_names = _get_track_names_spotify(artist_name)
    if not track_names:
        # Fallback: just search GetSongBPM directly with artist name
        track_names = [artist_name]

    # Look up keys on GetSongBPM
    semitones = []
    for title in track_names:
        key = _lookup_song_key(songbpm_key, title, artist_name)
        if key is not None:
            semitones.append(key)
        if len(semitones) >= 5:
            break

    if not semitones:
        return "", 0

    # Circular average to handle wrapping (B->C)
    angles = [s * 2 * math.pi / 12 for s in semitones]
    avg_sin = sum(math.sin(a) for a in angles) / len(angles)
    avg_cos = sum(math.cos(a) for a in angles) / len(angles)
    avg_angle = math.atan2(avg_sin, avg_cos)
    if avg_angle < 0:
        avg_angle += 2 * math.pi
    avg_semi = round(avg_angle * 12 / (2 * math.pi)) % 12

    note = SEMI_TO_NAME[avg_semi]
    hz = C4_HZ * (2 ** (avg_semi / 12))
    return f"{note}4", hz


def _clean_title(title: str) -> str:
    """Strip feat/remix/remaster suffixes for better search matching."""
    import re
    # Remove parenthetical and bracketed suffixes (feat., remix, remaster, etc.)
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)
    return title.strip()


def _normalize_name(name: str) -> str:
    """Strip accents and lowercase for fuzzy artist matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _get_track_names_spotify(artist_name: str) -> list[str]:
    """Get track names for an artist via Spotify search (no auth-restricted APIs)."""
    try:
        from services.spotify_client import SpotifyClient
        client = SpotifyClient()
        token = client._get_token()
        if not token:
            return []
        resp = requests.get(
            client.SEARCH_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"q": f"artist:{artist_name}", "type": "track", "limit": 10},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        tracks = resp.json().get("tracks", {}).get("items", [])
        # Filter to tracks actually by this artist, clean titles
        target = _normalize_name(artist_name)
        names = []
        for t in tracks:
            artists = [_normalize_name(a["name"]) for a in t.get("artists", [])]
            if target in artists:
                names.append(_clean_title(t["name"]))
        return names
    except Exception:
        return []


def _lookup_song_key(api_key: str, song_title: str, artist_name: str) -> int | None:
    """Look up a song's key on GetSongBPM using 'both' search."""
    try:
        resp = requests.get(
            "https://api.getsong.co/search/",
            params={
                "type": "both",
                "lookup": f"song:{song_title} artist:{artist_name}",
                "api_key": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("search", [])
        if not isinstance(results, list) or not results:
            return None
        # Take the first result that matches the artist
        for song in results:
            a = song.get("artist", {}).get("name", "")
            if a.lower() == artist_name.lower():
                key_of = song.get("key_of", "")
                if key_of:
                    return _parse_key(key_of)
        # If no exact match, use first result's key
        key_of = results[0].get("key_of", "")
        if key_of:
            return _parse_key(key_of)
    except Exception:
        pass
    return None
