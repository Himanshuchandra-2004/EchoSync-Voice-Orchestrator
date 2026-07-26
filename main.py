"""
EchoSync Voice Orchestrator

Browser mic audio --> FastAPI WebSocket --> Groq Whisper (STT)
                                        --> Groq (LLM, streamed, sentence-chunked)
                                        --> Edge TTS (streamed)
                                        --> back to browser for playback

Run with:  uvicorn main:app --reload
"""
import asyncio
import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from whisper_stt import WhisperSTT
from edge_tts_wrapper import EdgeTTS
from groq_llm import GroqLLM

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-agent")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("client connected")

    connected = True
    send_lock = asyncio.Lock()

    async def send_json(payload: dict):
        if not connected:
            return
        try:
            async with send_lock:
                await websocket.send_text(json.dumps(payload))
        except RuntimeError:
            pass  # connection already closed

    async def send_audio(chunk: bytes):
        if not connected:
            return
        try:
            async with send_lock:
                await websocket.send_bytes(chunk)
        except RuntimeError:
            pass  # connection already closed

    llm = GroqLLM()
    tts = EdgeTTS()

    # Guards against overlapping replies if the user talks again before
    # the agent finishes speaking.
    reply_lock = asyncio.Lock()

    async def handle_final_transcript(user_text: str, detected_lang: str = "en"):
        async with reply_lock:
            await send_json({"type": "user_transcript", "text": user_text, "final": True})
            await send_json({"type": "agent_start"})
            try:
                # Pipeline: LLM generates sentences into a queue while TTS
                # consumes them concurrently.
                sentence_queue = asyncio.Queue()

                async def produce():
                    try:
                        async for sentence in llm.stream_reply(user_text, detected_lang=detected_lang):
                            await send_json({"type": "agent_text", "text": sentence})
                            await sentence_queue.put(sentence)
                    except Exception:
                        log.exception("LLM error")
                    finally:
                        await sentence_queue.put(None)  # sentinel

                async def consume():
                    while True:
                        sentence = await sentence_queue.get()
                        if sentence is None:
                            break
                        await send_json({"type": "audio_start"})
                        async for audio_chunk in tts.stream_sentence(sentence, lang=detected_lang):
                            await send_audio(audio_chunk)
                        await send_json({"type": "audio_end"})

                producer = asyncio.create_task(produce())
                await consume()
                await producer

            except Exception:
                log.exception("error generating reply")
                await send_json({"type": "error", "text": "Sorry, something went wrong."})
            await send_json({"type": "agent_end"})

    async def on_transcript(text: str, is_final: bool, detected_lang: str = "en"):
        if is_final:
            asyncio.create_task(handle_final_transcript(text, detected_lang))
        else:
            await send_json({"type": "user_transcript", "text": text, "final": False})

    stt = WhisperSTT(on_transcript=on_transcript)
    await stt.connect()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                await stt.send_audio(message["bytes"])
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "flush":
                        await stt.flush()
                except (json.JSONDecodeError, AttributeError):
                    pass
            elif message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        log.info("client disconnected")
    finally:
        connected = False
        await stt.close()
        await tts.close()
