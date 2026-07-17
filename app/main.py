"""Orchestrator entrypoint (BLUEPRINT F-01, F-04, F-06..F-10, R-01..R-11).

Reliability model: seed every task with Tier-4 template captions and write a valid,
complete results.json at t=0, then overwrite each task's captions as real ones land.
Consequences:
  * OUTPUT_MISSING is impossible after startup (F-06).
  * Every requested style is always present (R-10, F-02).
  * A crash/timeout at any moment leaves a valid, complete file (F-07, F-08).
The process exits 0 in every recoverable path (R-03).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

import httpx

from . import media as media_mod
from . import schema, styles
from .backends import build_registry, ordered_backends
from .config import Config
from .pipeline import caption_clip

log = logging.getLogger("stylecap")


def _load_tasks(path: str) -> list[dict]:
    """Parse /input/tasks.json tolerantly (F-01, R-20)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [data])
    if not isinstance(data, list):
        raise ValueError("tasks.json is not a list")
    all_styles = list(styles.STYLE_DEFS.keys())
    tasks: list[dict] = []
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            continue
        tid = t.get("task_id") or f"task_{i}"
        stl = t.get("styles") or all_styles
        stl = [str(s) for s in stl if isinstance(s, str) and s.strip()] or all_styles
        tasks.append({"task_id": str(tid), "video_url": str(t.get("video_url", "") or ""),
                      "styles": stl})
    return tasks


def _atomic_write(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".results.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _serialize(results: dict, tasks: list[dict]) -> list[dict]:
    out = []
    for t in tasks:
        entry = {"task_id": t["task_id"], "captions": results[t["task_id"]]}
        if Config.DEBUG_CONTEXT and t.get("_context"):   # local-debug only; off for submissions
            entry["context"] = t["_context"]
            if t.get("_pass1"):
                entry["pass1_captions"] = t["_pass1"]
        out.append(entry)
    return out


def _templates_for(task: dict) -> dict:
    return {s: styles.template_caption(s) for s in task["styles"]}


async def run() -> None:
    log.info("StyleCap start | %s", Config.summary())
    tasks = _load_tasks(Config.INPUT_PATH)
    log.info("loaded %d tasks", len(tasks))

    # Seed valid, complete output immediately (F-06/F-07 insurance).
    results: dict[str, dict] = {t["task_id"]: _templates_for(t) for t in tasks}
    lock = asyncio.Lock()
    _atomic_write(Config.OUTPUT_PATH, _serialize(results, tasks))

    client = httpx.AsyncClient()
    reg = build_registry(client)
    backends = ordered_backends(reg)
    log.info("backends available: %s", [b.name for b in backends] or "NONE (template-only)")

    sem = asyncio.Semaphore(Config.CONCURRENCY)

    async def process(task: dict) -> None:
        tid = task["task_id"]
        t0 = time.monotonic()
        used = "template"
        try:
            if not task["video_url"]:
                raise RuntimeError("missing video_url")
            if Config.MODE == "native_video":
                # The model watches the URL directly — no local download/extract.
                caps, used = await caption_clip(task, None, reg, Config.MODE)
            else:
                m = await media_mod.fetch_and_extract(
                    tid, task["video_url"], client, Config.PER_CLIP_TIMEOUT_S)
                caps, used = await caption_clip(task, m, reg, Config.MODE)
        except Exception as e:  # F-05/F-08: any failure -> keep templates
            log.warning("[%s] failed (%s) -> templates", tid, e)
            caps = _templates_for(task)
        finally:
            media_mod.cleanup(tid)
        async with lock:
            results[tid] = caps
            _atomic_write(Config.OUTPUT_PATH, _serialize(results, tasks))  # F-06 incremental
        log.info("[%s] done via %s in %.1fs", tid, used, time.monotonic() - t0)

    async def guarded(task: dict) -> None:
        async with sem:
            try:
                await asyncio.wait_for(process(task), timeout=Config.PER_CLIP_TIMEOUT_S)  # F-04
            except asyncio.TimeoutError:
                log.warning("[%s] per-clip timeout; template stands", task["task_id"])

    try:
        await asyncio.wait_for(
            asyncio.gather(*(guarded(t) for t in tasks), return_exceptions=True),
            timeout=Config.WATCHDOG_S)  # F-07
    except asyncio.TimeoutError:
        log.warning("global watchdog fired at %ss", Config.WATCHDOG_S)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass

    # Final self-check + repair (F-09).
    final = _serialize(results, tasks)
    problems = schema.validate_results(final, tasks)
    if problems:
        log.error("schema repair needed: %s", problems[:8])
        for t in tasks:
            caps = results.get(t["task_id"]) or {}
            for s in t["styles"]:
                if not caps.get(s):
                    caps[s] = styles.template_caption(s)
            results[t["task_id"]] = caps
    _atomic_write(Config.OUTPUT_PATH, _serialize(results, tasks))
    log.info("StyleCap wrote %d results -> %s", len(tasks), Config.OUTPUT_PATH)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    try:
        asyncio.run(run())
    except Exception as e:  # last-ditch emergency write (F-08)
        log.exception("fatal: %s", e)
        try:
            tasks = _load_tasks(Config.INPUT_PATH)
            _atomic_write(Config.OUTPUT_PATH,
                          [{"task_id": t["task_id"], "captions": _templates_for(t)} for t in tasks])
        except Exception:
            pass
    return 0  # R-03: always exit 0


if __name__ == "__main__":
    sys.exit(main())
