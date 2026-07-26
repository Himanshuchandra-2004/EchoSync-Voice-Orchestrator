"""
Streams a reply from Groq (OpenAI-compatible endpoint) and yields it
sentence-by-sentence, so the caller can start TTS on sentence 1 while
sentence 2 is still being generated.
"""
import os
import re

from openai import AsyncOpenAI

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # also split on Devanagari danda ।

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
}

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant on a live phone-style call. "
    "Keep replies short and conversational -- 1-3 sentences -- since they will "
    "be spoken aloud. Avoid lists, markdown, or anything that only makes sense "
    "in writing. "
    "IMPORTANT: Reply in the SAME language the user speaks. If the user speaks "
    "Hindi, reply in Hindi. If they speak Kannada, reply in Kannada. If they "
    "speak Marathi, reply in Marathi. If they speak English, reply in English. "
    "Match the user's language exactly."
)


class GroqLLM:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self._model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def stream_reply(self, user_text: str, detected_lang: str = "en"):
        """
        Async generator. Yields complete sentences as soon as they're ready,
        then yields any trailing partial text at the end. Also updates
        internal conversation history for context.
        """
        # Prepend a language instruction so the LLM replies in the right language
        lang_name = LANG_NAMES.get(detected_lang, "English")
        if detected_lang != "en":
            tagged_text = f"[The user is speaking in {lang_name}. You MUST reply in {lang_name} only.]\n{user_text}"
        else:
            tagged_text = user_text

        self._history.append({"role": "user", "content": tagged_text})

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            stream=True,
            temperature=0.7,
        )

        buffer = ""
        full_reply = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buffer += delta
            full_reply += delta

            parts = SENTENCE_END.split(buffer)
            # Keep the last (possibly incomplete) part in the buffer;
            # yield everything before it.
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    yield sentence
            buffer = parts[-1]

        if buffer.strip():
            yield buffer.strip()

        self._history.append({"role": "assistant", "content": full_reply})
