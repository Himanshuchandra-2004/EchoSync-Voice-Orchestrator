"""
Thin wrapper around Deepgram's live streaming transcription WebSocket.

Deepgram's streaming API is itself a WebSocket, so this class holds a
second WebSocket connection (browser <-> our server <-> Deepgram) and
exposes two simple async methods: send_audio() and a background task
that reads transcripts and calls a callback.
"""
import asyncio
import json
import logging
import os

import websockets

log = logging.getLogger("voice-agent")

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&language=multi"               # auto-detect Hindi, English, etc.
    "&interim_results=true"
    "&endpointing=200"              # faster utterance finalization (was 300)
    "&smart_format=true"
    "&utterance_end_ms=1000"        # faster end-of-speech detection
)


class DeepgramSTT:
    def __init__(self, on_transcript):
        """
        on_transcript: async callback(text: str, is_final: bool) called
        every time Deepgram emits a transcript chunk.
        """
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self._api_key = api_key
        self._on_transcript = on_transcript
        self._ws = None
        self._reader_task = None

    async def connect(self):
        self._ws = await websockets.connect(
            DEEPGRAM_URL,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            ping_interval=5,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def send_audio(self, pcm16_bytes: bytes):
        if self._ws is not None:
            await self._ws.send(pcm16_bytes)

    async def close(self):
        if self._ws is not None:
            # Tells Deepgram we're done sending audio so it flushes the final transcript.
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            await self._ws.close()
        if self._reader_task is not None:
            self._reader_task.cancel()

    async def _read_loop(self):
        try:
            async for message in self._ws:
                data = json.loads(message)
                if data.get("type") != "Results":
                    continue
                channel = data["channel"]
                alt = channel["alternatives"][0]
                text = alt.get("transcript", "")
                if not text:
                    continue
                is_final = data.get("is_final", False) and data.get("speech_final", False)
                detected_lang = channel.get("detected_language", "en")
                log.info("STT [lang=%s final=%s]: %s", detected_lang, is_final, text)
                await self._on_transcript(text, is_final, detected_lang)
        except websockets.exceptions.ConnectionClosed:
            pass
