"""
Streams TTS audio using edge-tts (Microsoft Edge's TTS engine).
Completely free, no API key required, good quality voices.
Auto-selects voice based on detected language from STT.
"""
import re
import edge_tts

# Language code → edge-tts voice mapping
VOICE_MAP = {
    "en": "en-US-AndrewNeural",
    "hi": "hi-IN-MadhurNeural",
    "kn": "kn-IN-GaganNeural",
    "mr": "mr-IN-ManoharNeural",
}
DEFAULT_VOICE = "en-US-AndrewNeural"

# Fallback: detect Kannada script (separate from Devanagari)
_KANNADA = re.compile(r"[\u0C80-\u0CFF]")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


class EdgeTTS:
    def __init__(self):
        pass

    def _pick_voice(self, text: str, lang: str | None = None) -> str:
        """Pick voice from STT-detected language, falling back to script detection."""
        # If STT provided a language code, use it directly
        if lang and lang in VOICE_MAP:
            return VOICE_MAP[lang]

        # Fallback: detect by script
        if _KANNADA.search(text):
            return VOICE_MAP["kn"]
        if _DEVANAGARI.search(text):
            return VOICE_MAP["hi"]       # works for Hindi & Marathi fallback
        return DEFAULT_VOICE

    async def stream_sentence(self, text: str, lang: str | None = None):
        """Async generator yielding raw mp3 audio bytes in chunks."""
        voice = self._pick_voice(text, lang)
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate="+30%",      # faster speech for snappier voice
            pitch="+0Hz",
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def close(self):
        pass  # edge-tts doesn't hold persistent connections
