"""
Streams TTS audio from ElevenLabs for one sentence at a time.
Uses the HTTP streaming endpoint (simpler than their websocket API and
plenty fast for sentence-level chunks).
"""
import os

import httpx

BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsTTS:
    def __init__(self):
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        self._api_key = api_key
        self._voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def stream_sentence(self, text: str):
        """Async generator yielding raw mp3 audio bytes in chunks."""
        url = f"{BASE_URL}/{self._voice_id}/stream"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_flash_v2_5",  # low-latency model
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        async with self._client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk

    async def close(self):
        await self._client.aclose()
