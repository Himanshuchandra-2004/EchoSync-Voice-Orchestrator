"""
Speech-to-text using Groq's Whisper endpoint.
Supports Kannada, Hindi, Marathi, English and 50+ other languages.
Uses simple energy-based VAD to detect end of speech before sending to Whisper.
"""
import asyncio
import io
import logging
import os
import struct
import wave

import httpx

log = logging.getLogger("voice-agent")

WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Whisper returns full language names; normalize to ISO 639-1 codes
# that the rest of the pipeline (LLM + TTS) expects.
LANG_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "marathi": "mr",
    "urdu": "hi",       # spoken Urdu ≈ Hindi; use Hindi voice & prompt
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "malayalam": "ml",
}


class WhisperSTT:
    def __init__(self, on_transcript):
        """
        on_transcript: async callback(text: str, is_final: bool, detected_lang: str)
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._api_key = api_key
        self._on_transcript = on_transcript
        self._client = httpx.AsyncClient(timeout=30.0)
        self._buffer = bytearray()
        self._silence_frames = 0
        self._speech_frames = 0
        self._speaking = False
        self._sent_listening = False

        # Audio params (must match browser's downsampled output)
        self._sample_rate = 16000
        self._channels = 1
        self._sample_width = 2  # 16-bit PCM

        # VAD tuning
        # Each audio chunk from the browser is ~85ms (4096 samples at 48kHz → ~1365 at 16kHz)
        self._silence_threshold = 400   # RMS energy; adjust if too sensitive / not sensitive enough
        self._silence_max_frames = 5    # ~420ms of silence triggers end-of-speech (was 10)
        self._min_speech_frames = 2     # ~170ms minimum speech to avoid transcribing noise (was 4)

    async def connect(self):
        """No persistent connection needed for Whisper."""
        log.info("WhisperSTT ready (using Groq whisper-large-v3-turbo)")

    async def flush(self):
        """Browser detected silence — transcribe immediately without waiting for server VAD."""
        if len(self._buffer) > 0 and self._speech_frames >= self._min_speech_frames:
            audio_data = bytes(self._buffer)
            self._buffer = bytearray()
            self._speaking = False
            self._silence_frames = 0
            self._speech_frames = 0
            self._sent_listening = False
            asyncio.create_task(self._transcribe(audio_data))

    async def send_audio(self, pcm16_bytes: bytes):
        """Buffer incoming audio. Detect speech boundaries using energy-based VAD."""
        rms = self._calculate_rms(pcm16_bytes)

        if rms > self._silence_threshold:
            # Speech detected
            if not self._speaking:
                self._speaking = True
                self._silence_frames = 0
                self._speech_frames = 0
                self._sent_listening = False

            self._speech_frames += 1
            self._silence_frames = 0
            self._buffer.extend(pcm16_bytes)

            # Show "listening" indicator after a few frames of confirmed speech
            if self._speech_frames >= 2 and not self._sent_listening:
                self._sent_listening = True
                await self._on_transcript("…", False, "en")

        elif self._speaking:
            # Silence during speech — might be end of utterance
            self._silence_frames += 1
            self._buffer.extend(pcm16_bytes)  # keep buffering during short pauses

            if self._silence_frames >= self._silence_max_frames:
                # End of speech detected
                audio_data = bytes(self._buffer)
                speech_frames = self._speech_frames
                self._buffer = bytearray()
                self._speaking = False
                self._silence_frames = 0
                self._speech_frames = 0
                self._sent_listening = False

                if speech_frames >= self._min_speech_frames:
                    asyncio.create_task(self._transcribe(audio_data))

    def _calculate_rms(self, pcm16_bytes: bytes) -> float:
        """Root mean square energy of a PCM16 audio chunk."""
        n = len(pcm16_bytes) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", pcm16_bytes[:n * 2])
        return (sum(s * s for s in samples) / n) ** 0.5

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Wrap raw PCM16 data in a WAV container for Whisper."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(self._sample_width)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    async def _transcribe(self, pcm_data: bytes):
        """Send buffered audio to Groq Whisper, invoke callback with result."""
        try:
            wav_data = self._pcm_to_wav(pcm_data)
            log.info("Sending %.1f sec of audio to Whisper…", len(pcm_data) / (self._sample_rate * self._sample_width))

            resp = await self._client.post(
                WHISPER_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                },
            )
            resp.raise_for_status()
            result = resp.json()

            text = result.get("text", "").strip()
            raw_lang = result.get("language", "english")
            detected_lang = LANG_TO_CODE.get(raw_lang.lower(), "en")

            log.info("STT [lang=%s (%s)]: %s", detected_lang, raw_lang, text)

            if text:
                await self._on_transcript(text, True, detected_lang)

        except Exception:
            log.exception("Whisper transcription error")

    async def close(self):
        await self._client.aclose()
