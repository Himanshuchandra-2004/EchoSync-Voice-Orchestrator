"""
Streams TTS audio using Lokutor's WebSocket API.
Low-latency (~139ms TTFB), persistent connection, raw PCM16 output.
Designed for real-time voice agent pipelines.

Audio format: Int16 PCM, 44.1 kHz, mono
WebSocket: wss://api.lokutor.com/ws/tts
"""
import json
import logging
import os

import websockets

log = logging.getLogger("lokutor-tts")

# Lokutor voices — pick a good default
DEFAULT_VOICE = "M1"
DEFAULT_LANGUAGE = "en"
DEFAULT_SPEED = 1.05
DEFAULT_STEPS = 8  # ultra-fast for conversational latency


class LokutorTTS:
    def __init__(self):
        api_key = os.environ.get("LOKUTOR_API_KEY")
        if not api_key:
            raise RuntimeError("LOKUTOR_API_KEY is not set")
        self._api_key = api_key
        self._ws_url = f"wss://api.lokutor.com/ws/tts?api_key={api_key}"
        self._ws = None
        self._voice = os.environ.get("LOKUTOR_VOICE", DEFAULT_VOICE)
        self._language = os.environ.get("LOKUTOR_LANGUAGE", DEFAULT_LANGUAGE)
        self._speed = float(os.environ.get("LOKUTOR_SPEED", DEFAULT_SPEED))
        self._steps = int(os.environ.get("LOKUTOR_STEPS", DEFAULT_STEPS))

    async def _ensure_connection(self):
        """Open or reuse a persistent WebSocket connection."""
        if self._ws is None or self._ws.closed:
            log.info("connecting to Lokutor WebSocket…")
            self._ws = await websockets.connect(
                self._ws_url,
                max_size=2**20,  # 1 MB max frame
                ping_interval=20,
                ping_timeout=10,
            )
            log.info("Lokutor WebSocket connected")

    async def stream_sentence(self, text: str, lang: str | None = None):
        """
        Async generator yielding raw PCM16 audio bytes for a single sentence.

        The connection is kept open between calls so subsequent sentences
        avoid the handshake overhead (~50ms savings per sentence).

        Args:
            text: Text to synthesize
            lang: Language code (ignored for non-English since Lokutor
                  currently supports European languages only; defaults to 'en')
        """
        await self._ensure_connection()

        # Build the synthesis request
        request = {
            "text": text,
            "voice": self._voice,
            "language": lang if lang and lang in (
                "en", "es", "ca", "gl", "eu", "pt", "fr", "it", "de"
            ) else self._language,
            "speed": self._speed,
            "steps": self._steps,
        }

        try:
            await self._ws.send(json.dumps(request))

            async for message in self._ws:
                if isinstance(message, bytes):
                    # Raw PCM16 audio chunk
                    yield message
                elif isinstance(message, str):
                    if message == "EOS":
                        # End of synthesis for this sentence
                        break
                    else:
                        # Could be timing telemetry or error JSON
                        try:
                            data = json.loads(message)
                            if data.get("type") == "error":
                                err = data.get("data", {})
                                log.error(
                                    "Lokutor error: %s — %s",
                                    err.get("code", "unknown"),
                                    err.get("message", "no message"),
                                )
                                break
                            elif data.get("type") == "timing":
                                ttfb = data.get("ttfb_ms")
                                if ttfb is not None:
                                    log.info("Lokutor TTFB: %dms", ttfb)
                        except json.JSONDecodeError:
                            pass  # unknown text message, ignore

        except websockets.ConnectionClosed:
            log.warning("Lokutor WebSocket closed unexpectedly, will reconnect")
            self._ws = None
        except Exception:
            log.exception("Lokutor TTS error")
            self._ws = None

    async def close(self):
        """Close the persistent WebSocket connection."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
            log.info("Lokutor WebSocket closed")
        self._ws = None
