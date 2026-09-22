# EchoSync Voice Orchestrator

An ultra-low-latency, multilingual conversational voice agent built with FastAPI, Groq (Whisper STT + LLM), and Lokutor TTS (with Edge TTS fallback).

```
Browser Mic (16kHz PCM16)
       │
       ▼
FastAPI WebSocket Orchestrator
       │
       ├─► Groq Whisper Large v3 Turbo (STT, Multilingual with VAD)
       ├─► Groq LLM (Streamed & Sentence-Chunked)
       └─► Lokutor TTS (Persistent WebSocket, Raw PCM16 @ 44.1 kHz, ~200ms warm TTFB)
             [Fallback: Microsoft Edge TTS (Free, Multilingual MP3)]
       │
       ▼
Browser Speaker (Gapless PCM16 playback via Web Audio API)
```

---

## Features

- **Sub-Second Conversational Turnaround**: Pre-connects Lokutor's WebSocket on browser session start to eliminate cold-start handshake latency (~200ms warm TTFB to first audio).
- **Multilingual Support**: Supports English, Hindi (हिंदी), Kannada (ಕನ್ನಡ), Marathi (मराठी), and 50+ languages via Whisper STT and multilingual TTS.
- **Raw PCM16 Streaming**: Streams uncompressed 44.1 kHz Int16 PCM directly into the browser for immediate, gapless playback via the Web Audio API.
- **Concurrent Pipeline**: The LLM streams sentences into a queue while TTS synthesizes audio in parallel—audio starts playing while the LLM is still drafting subsequent sentences.
- **Resilient WebSockets**: Fully compatible with `websockets 13.x` and `14+`, using header-based authentication (`X-API-Key`).

---

## 1. Get API Keys

- **Groq API Key** (Required for Whisper STT and LLM): [console.groq.com/keys](https://console.groq.com/keys)
- **Lokutor API Key** (Required for primary low-latency PCM16 TTS): [lokutor.com](https://lokutor.com)
  *(If you don't have a Lokutor key, set `TTS_ENGINE=edge` in `.env` to use the free Microsoft Edge TTS engine).*

---

## 2. Setup

```bash
# Clone the repository
git clone https://github.com/Himanshuchandra-2004/EchoSync-Voice-Orchestrator.git
cd EchoSync-Voice-Orchestrator

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required
GROQ_API_KEY=gsk_...

# Primary TTS (Lokutor)
TTS_ENGINE=lokutor
LOKUTOR_API_KEY=lok_...

# Optional Lokutor Settings:
# LOKUTOR_VOICE=M1          # M1-M5, F1-F5
# LOKUTOR_LANGUAGE=en        # en, es, ca, gl, eu, pt, fr, it, de
# LOKUTOR_SPEED=1.05         # 0.5–2.0
# LOKUTOR_STEPS=8            # lower = faster (8=ultra-fast, 32=high quality)

# Or use free Edge TTS fallback:
# TTS_ENGINE=edge
```

---

## 3. Run

Start the server locally:

```bash
uvicorn main:app --reload
```

Open **`http://localhost:8000`** in your browser, click the microphone button, grant mic permissions, and speak naturally.

---

## Project Structure

```
├── main.py               # FastAPI server & WebSocket orchestrator
├── whisper_stt.py        # Groq Whisper Large v3 Turbo STT with client & server VAD
├── groq_llm.py           # Groq LLM streaming with sentence-boundary chunking
├── lokutor_tts.py        # Persistent WebSocket Lokutor TTS streaming PCM16 @ 44.1kHz
├── edge_tts_wrapper.py   # Fallback free Microsoft Edge TTS (MP3)
├── static/
│   └── index.html        # Web interface, mic capture (16kHz PCM16), gapless PCM scheduler
├── requirements.txt      # Project dependencies
└── .env.example          # Environment variable template
```

---

## Latency Optimizations

1. **Pre-warmed WebSocket**: TTS WebSocket connection is established immediately when the browser client connects, bypassing the initial ~900ms TLS/WS handshake before the user even finishes speaking.
2. **Streaming Sentence Pipeling**: Sentences are emitted as soon as sentence delimiters (`.`, `?`, `!`, newline) are reached, allowing TTS synthesis to begin while the remainder of the response is still generating.
3. **Web Audio PCM16 Buffer Scheduling**: Chunks of raw PCM audio are scheduled and scheduled gaplessly on the browser's `AudioContext` without needing container decoding (WAV/MP3).
4. **Header-based Auth**: Uses `X-API-Key` headers rather than URL query parameters to ensure API keys are secure and never exposed in logs.
