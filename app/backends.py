"""Model backends (BLUEPRINT F-05, F-11).

One OpenAI-compatible client serves Fireworks (serverless), the MI300X Gemma
endpoint (vLLM), and Gemini (OpenAI-compat surface). Best-of-N is done with N
concurrent requests so it works on any provider regardless of server-side `n`.

A backend never crashes the pipeline: on failure it raises ModelError and the
caller falls to the next tier. The "template" tier is not a backend here; the
pipeline uses styles.template_caption() when the model chain is exhausted.
"""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

from .config import Config

log = logging.getLogger("stylecap.backends")


class ModelError(Exception):
    pass


# HF's router is behind Cloudflare bot-fight-mode, which 403-challenges the default python-httpx
# User-Agent on heavy VISION traffic (large base64 image POSTs) — returning a challenge HTML page
# instead of routing to Gemma, silently dropping us to the Kimi fallback (and forfeiting the Gemma
# prize). A realistic browser UA passes the challenge. Harmless to the other providers; sent on all.
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _even_subset(items, k):
    if not k or len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


class OpenAICompatBackend:
    def __init__(self, name, base_url, api_key, model, client, supports_vision=True,
                 max_images=None, send_reasoning=True):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = client
        self.supports_vision = supports_vision
        self.max_images = max_images   # provider cap on images per request (e.g. HF Gemma = 5)
        self.send_reasoning = send_reasoning  # Anthropic's OpenAI-compat rejects reasoning_effort
        self._reasoning_unsupported = False   # set True if a model 400s on reasoning_effort
        self._json_unsupported = False        # set True if a model 400s on response_format

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": _BROWSER_UA}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _post_once(self, messages, temperature, max_tokens) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if Config.REASONING_EFFORT and self.send_reasoning and not self._reasoning_unsupported:
            payload["reasoning_effort"] = Config.REASONING_EFFORT
        if Config.JSON_MODE and not self._json_unsupported:
            payload["response_format"] = {"type": "json_object"}
        resp = await self.client.post(url, json=payload, headers=self._headers,
                                      timeout=Config.HTTP_TIMEOUT_S)
        if resp.status_code >= 400:
            body = resp.text[:300]
            # Some models reject reasoning_effort with a 400 — disable it and retry once.
            if resp.status_code == 400 and "reasoning_effort" in body and not self._reasoning_unsupported:
                self._reasoning_unsupported = True
                raise _Retry(f"{self.name} reasoning_effort unsupported; retrying without it")
            # Some models reject response_format — disable it and retry once.
            if resp.status_code == 400 and "response_format" in body and not self._json_unsupported:
                self._json_unsupported = True
                raise _Retry(f"{self.name} response_format unsupported; retrying without it")
            # 4xx (bad model/request) is not worth retrying; 5xx/429 is.
            retryable = resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500
            raise ModelError(f"{self.name} HTTP {resp.status_code}: {body}", ) if not retryable \
                else _Retry(f"{self.name} HTTP {resp.status_code}: {body}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"{self.name} malformed response: {e}: {str(data)[:200]}")
        # The account throttles by returning EMPTY completions (not HTTP 429) — retry, don't fall back.
        if not content.strip():
            raise _Retry(f"{self.name} empty completion (throttle?)")
        return content

    async def _call(self, messages, temperature, max_tokens) -> str:
        attempts = Config.HTTP_RETRIES + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                return await self._post_once(messages, temperature, max_tokens)
            except _Retry as e:
                last = e
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
            except ModelError:
                raise
            # backoff with jitter before the next attempt
            if i < attempts - 1:
                await asyncio.sleep(min(15.0, 2.0 * (2 ** i)) + random.uniform(0, 0.6))
        raise ModelError(f"{self.name} exhausted retries: {last}")

    @staticmethod
    def _vision_content(frames_b64, prompt) -> list:
        content = [{"type": "text", "text": prompt}]
        for f in frames_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
        return content

    async def chat_vision(self, frames_b64, prompt, *, temperature=0.4, max_tokens=700, n=1, prefill=""):
        if not self.supports_vision:
            raise ModelError(f"{self.name} has no vision support")
        frames_b64 = _even_subset(frames_b64, self.max_images)
        messages = [{"role": "user", "content": self._vision_content(frames_b64, prompt)}]
        if prefill:  # seed the assistant reply so a reasoning model goes straight to JSON (no ramble)
            messages.append({"role": "assistant", "content": prefill})
        return await self._many(messages, temperature, max_tokens, n)

    async def chat_text(self, prompt, *, temperature=0.9, max_tokens=400, n=1):
        messages = [{"role": "user", "content": prompt}]
        return await self._many(messages, temperature, max_tokens, n)

    async def chat_video(self, video_url, prompt, *, temperature=0.2, max_tokens=1200):
        """Native-video call: the model watches the actual clip (no frame sampling)."""
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        return await self._call([{"role": "user", "content": content}], temperature, max_tokens)

    async def _many(self, messages, temperature, max_tokens, n):
        if n <= 1:
            return await self._call(messages, temperature, max_tokens)
        results = await asyncio.gather(
            *[self._call(messages, temperature, max_tokens) for _ in range(n)],
            return_exceptions=True,
        )
        good = [r for r in results if isinstance(r, str) and r.strip()]
        if not good:
            raise ModelError(f"{self.name} all {n} samples failed")
        return good


class _Retry(Exception):
    pass


async def transcribe_audio(client: httpx.AsyncClient, audio_path: str) -> str:
    """Whisper via HF Inference Providers. Best-effort: returns '' on any failure."""
    if not (Config.USE_AUDIO and Config.HF_TOKEN and audio_path):
        return ""
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        if len(data) < 2048:  # lightweight VAD: skip near-silent/empty audio
            return ""
        url = f"https://router.huggingface.co/hf-inference/models/{Config.HF_ASR_MODEL}"
        r = await client.post(url, content=data, timeout=Config.HTTP_TIMEOUT_S,
                              headers={"Authorization": f"Bearer {Config.HF_TOKEN}",
                                       "Content-Type": "audio/wav"})
        if r.status_code == 200:
            return _clean_transcript((r.json().get("text") or "").strip())
        log.info("ASR HTTP %s: %s", r.status_code, r.text[:120])
    except Exception as e:
        log.info("ASR failed: %s", e)
    return ""


_ASR_HALLUCINATIONS = (
    "thanks for watching", "thank you for watching", "subscribe", "like and",
    "see you next", "music playing", "♪", "♫", "[music]", "(music)",
    "foreign", "bye-bye", "you're watching",
)


def _clean_transcript(text: str) -> str:
    """Guard against Whisper hallucinations on music/ambient audio (stock clips!).

    Whisper invents phrases like 'Thanks for watching!' on non-speech audio; feeding
    that into the fact-sheet poisons captions. Only keep transcripts that look like
    real speech.
    """
    t = text.strip()
    low = t.lower()
    if len(t.split()) < 4:
        return ""
    if any(h in low for h in _ASR_HALLUCINATIONS):
        return ""
    words = low.split()
    if len(set(words)) <= max(2, len(words) // 4):   # heavy repetition = junk
        return ""
    return t


def build_registry(client: httpx.AsyncClient) -> dict[str, OpenAICompatBackend]:
    """Instantiate every backend that has the credentials it needs (F-05)."""
    reg: dict[str, OpenAICompatBackend] = {}
    if Config.FIREWORKS_API_KEY:
        reg["fireworks"] = OpenAICompatBackend(
            "fireworks", Config.FIREWORKS_BASE_URL, Config.FIREWORKS_API_KEY,
            Config.FIREWORKS_VLM_MODEL, client, supports_vision=True)
    if Config.GEMMA_BASE_URL:
        reg["gemma"] = OpenAICompatBackend(
            "gemma", Config.GEMMA_BASE_URL, Config.GEMMA_API_KEY,
            Config.GEMMA_MODEL, client, supports_vision=True)
    if Config.GEMINI_API_KEY:
        # Gemini 3.x are thinking models that reject the OpenAI `reasoning_effort` param -> send_reasoning=False.
        reg["gemini"] = OpenAICompatBackend(
            "gemini", Config.GEMINI_BASE_URL, Config.GEMINI_API_KEY,
            Config.GEMINI_MODEL, client, supports_vision=True, send_reasoning=False)
    if Config.ANTHROPIC_API_KEY:
        # Claude via Anthropic's OpenAI-compat endpoint — the strong perceiver.
        reg["anthropic"] = OpenAICompatBackend(
            "anthropic", Config.ANTHROPIC_BASE_URL, Config.ANTHROPIC_API_KEY,
            Config.ANTHROPIC_MODEL, client, supports_vision=True, send_reasoning=False)
    if Config.HF_TOKEN:
        reg["hf"] = OpenAICompatBackend(
            "hf", Config.HF_BASE_URL, Config.HF_TOKEN,
            Config.HF_MODEL, client, supports_vision=True, max_images=Config.HF_MAX_IMAGES)
        if Config.HF2_MODEL:
            reg["hf2"] = OpenAICompatBackend(
                "hf2", Config.HF_BASE_URL, Config.HF_TOKEN,
                Config.HF2_MODEL, client, supports_vision=True, max_images=Config.HF_MAX_IMAGES)
    return reg


def ordered_backends(reg: dict[str, OpenAICompatBackend]) -> list[OpenAICompatBackend]:
    """Backends to try, in FALLBACK_CHAIN order, skipping 'template' and missing ones."""
    out = []
    for name in Config.FALLBACK_CHAIN:
        if name == "template":
            break
        b = reg.get(name)
        if b:
            out.append(b)
    return out
