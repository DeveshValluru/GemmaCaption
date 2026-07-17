"""Lean reference-description builder: ONE vision call per clip (Kimi, 16 frames).

Run inside the container (mount tests/eval as /data):
  docker run --rm --entrypoint python -e ... -v tests/eval:/data stylecap:dev6 /data/make_refs.py
Writes /data/refs.jsonl: {task_id, category, reference}
"""
import asyncio
import json
import sys

import httpx

from app import media as media_mod
from app.backends import OpenAICompatBackend
from app.config import Config

PROMPT = (
    "You are shown {n} frames sampled evenly from a short video clip, in order. "
    "Write a precise, factual reference description of the clip: main subject(s) and attributes, "
    "the sequence of actions, the setting, camera movement, and any on-screen text. "
    "5-7 sentences, plain prose, only what is clearly visible."
)


async def main():
    clips = json.load(open("/data/clips.json", encoding="utf-8"))
    client = httpx.AsyncClient()
    b = OpenAICompatBackend("ref", Config.HF_BASE_URL, Config.HF_TOKEN, Config.HF_MODEL,
                            client, supports_vision=True, max_images=16)
    out = []
    for i, c in enumerate(clips):
        tid = c.get("task_id", f"c{i}")
        try:
            m = await media_mod.fetch_and_extract(tid, c["url"], client, 150)
            ref = await b.chat_vision(m.send_frames, PROMPT.format(n=len(m.send_frames)),
                                      temperature=0.1, max_tokens=450, n=1)
            if ref and len(ref.strip()) > 80:
                out.append({"task_id": tid, "category": c.get("category", ""),
                            "reference": ref.strip()})
                print(f"[ref ok] {tid}", file=sys.stderr)
            else:
                print(f"[ref EMPTY] {tid}", file=sys.stderr)
        except Exception as e:
            print(f"[ref FAIL] {tid}: {e}", file=sys.stderr)
        finally:
            media_mod.cleanup(tid)
    with open("/data/refs.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)}/{len(clips)} refs", file=sys.stderr)
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
