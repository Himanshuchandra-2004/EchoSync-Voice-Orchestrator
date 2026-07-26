# EchoSync-Voice-Orchestrator

# Voice Agent Prototype

Browser mic → Deepgram (STT, streaming) → Groq (LLM, streamed) → ElevenLabs (TTS, streamed) → browser speaker.

No telephony yet — this runs entirely in your browser tab, so you can validate
whether the conversation *feels* good before adding Twilio.

## 1. Get free API keys

- **Deepgram**: https://console.deepgram.com → API Keys → Create a key
- **Groq**: https://console.groq.com/keys → Create API key
- **ElevenLabs**: https://elevenlabs.io → Profile → API keys

## 2. Set up the project

```bash
cd voice-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# now edit .env and paste in your three API keys
```

## 3. Run it

**Option A — Docker (recommended, no local Python setup needed):**

```bash
docker compose up --build
```

This builds the image and starts the server with live-reload — edits to
the `.py` files on your machine take effect without a rebuild. Stop
with `Ctrl+C`; `docker compose down` to remove the container.

**Option B — local Python:**

```bash
uvicorn main:app --reload
```

Open **http://localhost:8000**, click **Start**, allow mic access, and talk.
You should see your transcript appear live, and hear the agent's reply
start playing sentence-by-sentence (it doesn't wait for the whole reply
before speaking — that's the main latency trick).

## How it works

```
static/index.html   Captures mic audio, downsamples to 16kHz PCM16,
                     streams it to the server over a WebSocket. Plays
                     back TTS audio sentence-by-sentence as it arrives.

main.py              The WebSocket orchestrator. Wires the three services
                     together per connection.

deepgram_stt.py       Opens a second WebSocket to Deepgram's live
                     transcription API (model=nova-3, endpointing=300ms
                     for automatic turn detection). Emits interim and
                     final transcripts.

groq_llm.py           Streams a chat completion from Groq and yields
                     complete sentences as soon as they're ready — so
                     TTS can start on sentence 1 while the LLM is still
                     writing sentence 2.

elevenlabs_tts.py      Streams mp3 audio back for each sentence
                     (eleven_flash_v2_5 — their lowest-latency model).
```

## Known rough edges (it's a prototype)

- **Endpointing** (deciding when the user has finished talking) uses
  Deepgram's built-in 300ms silence timer. It's fine for testing but
  will sometimes cut you off mid-thought or wait too long on filler
  words. Deepgram's separate **Flux** model is built specifically for
  this and is worth trying once the basic loop feels right.
- **No barge-in**: if you start talking while the agent is still
  speaking, the current code will queue a second reply rather than
  interrupting the first. Worth fixing once you're happy with the
  base latency.
- **ScriptProcessorNode** is technically deprecated in favor of
  AudioWorklet, but it's simpler to read and still works in every
  major browser — fine for a prototype.

## Next steps once this feels good

1. Swap in **Cartesia Sonic** for TTS if you want to compare latency/voice quality.
2. Add **Twilio Media Streams** so it works over a real phone call —
   the STT/LLM/TTS pipeline in `main.py` barely changes, you just swap
   the WebSocket source from the browser to Twilio's audio stream
   (and handle μ-law 8kHz instead of PCM16 16kHz).
3. Add barge-in / interruption handling.

