"""Teacher-data generation for distillation.

Gemma 4 31B (via HF) is the TEACHER: for each clip it produces a verified
fact-sheet + one caption per style, exactly as the runtime twostage pipeline does.
Output is a JSONL of teacher records; a later step converts it to TRL chat examples
to LoRA-distill a smaller Gemma (12B / E4B) that punches above its weight.

Run inside the container (has ffmpeg):
    docker run --rm --entrypoint python \
      -e HF_TOKEN=... -v <clips_dir>:/data stylecap:v2 \
      -m app.datagen /data/clips.json /data/teacher.jsonl

clips.json: [{"url": "...", "category": "animals"}, ...]
Each output line: {url, category, frames_b64[], factsheet, captions{style:cap}}
"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx

from . import media as media_mod
from . import styles
from .backends import OpenAICompatBackend
from .config import Config
from .pipeline import _perceive, _stylize_one


async def _gen_one(backend, client, clip, i, sem):
    url = clip["url"]
    cat = clip.get("category", "")
    tid = f"td{i}"
    async with sem:
        try:
            m = await media_mod.fetch_and_extract(tid, url, client, 150)
            factsheet = await _perceive(backend, m)
            if not factsheet:
                return None
            caps: dict = {}
            prev: list = []
            for st in styles.STYLE_DEFS:  # all 4 styles
                c = await _stylize_one(backend, factsheet, st, prev)
                if c:
                    caps[st] = c
                    prev.append((st, c))
            if len(caps) < len(styles.STYLE_DEFS):
                return None
            print(f"[ok {i}] {cat:12s} {url.split('/')[-1][:40]}", file=sys.stderr)
            return {"url": url, "category": cat,
                    "frames_b64": m.send_frames, "factsheet": factsheet, "captions": caps}
        except Exception as e:
            print(f"[skip {i}] {url}: {e}", file=sys.stderr)
            return None
        finally:
            media_mod.cleanup(tid)


async def main():
    clips = json.load(open(sys.argv[1], encoding="utf-8"))
    out_path = sys.argv[2]
    if not Config.HF_TOKEN:
        print("HF_TOKEN required", file=sys.stderr)
        sys.exit(1)

    client = httpx.AsyncClient()
    teacher = OpenAICompatBackend("hf", Config.HF_BASE_URL, Config.HF_TOKEN,
                                  Config.HF_MODEL, client, supports_vision=True,
                                  max_images=Config.HF_MAX_IMAGES)
    sem = asyncio.Semaphore(Config.CONCURRENCY)
    results = await asyncio.gather(*(_gen_one(teacher, client, c, i, sem)
                                     for i, c in enumerate(clips)))
    await client.aclose()

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n}/{len(clips)} teacher records -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
