# GetSongBPM API Reference

Base URL: `https://api.getsong.co`  
Rate Limit: 3000 requests/hour  
Auth: API key passed as `api_key` query parameter  
Requirement: Backlink to getsongbpm.com (added in Settings page)

---

## Authentication

All requests require `api_key` as a query parameter:

```
https://api.getsong.co/{endpoint}/?api_key=YOUR_API_KEY&...
```

---

## Endpoints

### 1. Search

Search for artists or songs by name.

**URL:** `GET /search/`

**Parameters:**
| Param    | Type   | Required | Description                        |
|----------|--------|----------|------------------------------------|
| api_key  | string | yes      | Your API key                       |
| type     | string | yes      | `artist` or `song`                 |
| lookup   | string | yes      | Search query (artist or song name) |

**Example:**
```
GET https://api.getsong.co/search/?api_key=KEY&type=artist&lookup=drake
GET https://api.getsong.co/search/?api_key=KEY&type=song&lookup=hotline+bling
```

**Response (artist search):**
```json
{
  "search": [
    {
      "id": "3wz1Yz",
      "display_name": "Drake",
      "img": "https://..."
    }
  ]
}
```

**Response (song search):**
```json
{
  "search": [
    {
      "id": "abc123",
      "title": "Hotline Bling",
      "artist": {
        "id": "3wz1Yz",
        "name": "Drake"
      }
    }
  ]
}
```

---

### 2. Song Details

Get full details for a song including BPM, key, and metadata.

**URL:** `GET /song/`

**Parameters:**
| Param    | Type   | Required | Description       |
|----------|--------|----------|-------------------|
| api_key  | string | yes      | Your API key      |
| id       | string | yes      | Song ID from search |

**Example:**
```
GET https://api.getsong.co/song/?api_key=KEY&id=abc123
```

**Response:**
```json
{
  "song": {
    "id": "abc123",
    "title": "Hotline Bling",
    "uri": "https://getsongbpm.com/song/hotline-bling/abc123",
    "tempo": 135,
    "time_sig": "4/4",
    "artist": {
      "id": "3wz1Yz",
      "name": "Drake",
      "uri": "https://getsongbpm.com/artist/drake/3wz1Yz",
      "img": "https://..."
    },
    "album": {
      "title": "Views",
      "uri": "https://getsongbpm.com/album/views/xyz789"
    },
    "music": {
      "key_of": "F",
      "mode": "Major",
      "key_aliases": []
    },
    "open_key": "7d",
    "music_video": {
      "youtube": "https://youtube.com/..."
    }
  }
}
```

**Key fields we use:**
- `song.music.key_of` — The musical key (e.g. "F", "C#", "Bb")
- `song.music.mode` — "Major" or "Minor"
- `song.tempo` — BPM as integer

---

### 3. Artist Details

Get artist info.

**URL:** `GET /artist/`

**Parameters:**
| Param    | Type   | Required | Description          |
|----------|--------|----------|----------------------|
| api_key  | string | yes      | Your API key         |
| id       | string | yes      | Artist ID from search |

**Example:**
```
GET https://api.getsong.co/artist/?api_key=KEY&id=3wz1Yz
```

**Response:**
```json
{
  "artist": {
    "id": "3wz1Yz",
    "display_name": "Drake",
    "uri": "https://getsongbpm.com/artist/drake/3wz1Yz",
    "img": "https://...",
    "genres": ["hip hop", "rap", "pop"],
    "from": "Toronto, Canada",
    "mbid": "musicbrainz-id"
  }
}
```

---

### 4. Artist's Songs

Get list of songs by an artist.

**URL:** `GET /artist/songs/`

**Parameters:**
| Param    | Type   | Required | Description          |
|----------|--------|----------|----------------------|
| api_key  | string | yes      | Your API key         |
| id       | string | yes      | Artist ID from search |

**Example:**
```
GET https://api.getsong.co/artist/songs/?api_key=KEY&id=3wz1Yz
```

**Response:**
```json
{
  "songs": [
    {
      "id": "abc123",
      "title": "Hotline Bling",
      "uri": "https://getsongbpm.com/song/hotline-bling/abc123",
      "artist": {
        "id": "3wz1Yz",
        "name": "Drake"
      }
    },
    {
      "id": "def456",
      "title": "God's Plan",
      "uri": "https://getsongbpm.com/song/gods-plan/def456",
      "artist": {
        "id": "3wz1Yz",
        "name": "Drake"
      }
    }
  ]
}
```

---

### 5. Tempo/Key Lookup

Search songs by tempo or key.

**URL:** `GET /tempo/`

**Parameters:**
| Param    | Type   | Required | Description                      |
|----------|--------|----------|----------------------------------|
| api_key  | string | yes      | Your API key                     |
| bpm      | int    | yes      | Beats per minute to search for   |

**URL:** `GET /key/`

**Parameters:**
| Param    | Type   | Required | Description              |
|----------|--------|----------|--------------------------|
| api_key  | string | yes      | Your API key             |
| key_of   | string | yes      | Musical key (e.g. "C#")  |

---

## Our Usage Flow (SomerSVC)

When estimating an artist's vocal center for smart transpose:

1. **Search artist:** `GET /search/?type=artist&lookup=ARTIST_NAME`
2. **Get songs:** `GET /artist/songs/?id=ARTIST_ID` (take top 5)
3. **Get each song's key:** `GET /song/?id=SONG_ID` (read `music.key_of`)
4. **Average the keys:** Circular average of semitone values to find the artist's typical key center
5. **Map to Hz:** Convert the averaged note to a frequency for transpose calculations

This gives us real musical key data instead of guessing from genres.

---

## Notes

- Free tier: 3000 requests per hour (plenty for our use)
- API key is stored in `~/.svc-gui/config.json` as `songbpm_api_key`
- Attribution link required: "Song key data powered by GetSongBPM.com" (in Settings page)
- Keys come as note names: "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
- Minor keys may have "m" suffix (e.g. "Am")
- `open_key` field uses Camelot/Open Key notation (e.g. "7d") — not used by us currently
