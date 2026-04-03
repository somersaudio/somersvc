"""Spotify API client for fetching artist images."""

import base64
import os
import requests
from pathlib import Path


class SpotifyClient:
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    SEARCH_URL = "https://api.spotify.com/v1/search"

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id or os.environ.get("SOMERSVC_SPOTIFY_ID", "")
        self.client_secret = client_secret or os.environ.get("SOMERSVC_SPOTIFY_SECRET", "")
        self._token = ""

    def _get_token(self) -> str:
        if self._token:
            return self._token
        if not self.client_id or not self.client_secret:
            return ""
        try:
            auth = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            resp = requests.post(
                self.TOKEN_URL,
                headers={"Authorization": f"Basic {auth}"},
                data={"grant_type": "client_credentials"},
                timeout=10,
            )
            if resp.status_code == 200:
                self._token = resp.json().get("access_token", "")
                return self._token
        except Exception:
            pass
        return ""

    def search_artist_image(self, artist_name: str) -> str | None:
        """Search Spotify for an artist and return their image URL."""
        token = self._get_token()
        if not token:
            return None
        try:
            resp = requests.get(
                self.SEARCH_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={"q": artist_name, "type": "artist", "limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                artists = resp.json().get("artists", {}).get("items", [])
                if artists and artists[0].get("images"):
                    # Return the medium-sized image (usually index 1)
                    images = artists[0]["images"]
                    if len(images) > 1:
                        return images[1]["url"]
                    return images[0]["url"]
        except Exception:
            pass
        return None

    def download_artist_image(self, artist_name: str, save_path: str) -> bool:
        """Download artist image and save to disk."""
        url = self.search_artist_image(artist_name)
        if not url:
            return False
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
        return False
