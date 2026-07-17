"""Media ingest: download + single-pass ffmpeg frames & audio (BLUEPRINT F-03, N-02, N-03).

One decode pass per clip yields evenly-spaced frames spanning the WHOLE clip (so a
2-minute video is not truncated to its first 60s) plus optional mono audio. Nothing
here raises to the caller on partial failure except a hard "no frames at all", which
the pipeline treats as a Tier-4 clip.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil

import httpx
from PIL import Image

from .config import Config

log = logging.getLogger("stylecap.media")


class Media:
    def __init__(self, all_frames, send_frames, audio_path, duration, has_audio, tile="",
                 n_slow=0, context_frames=None, gemma_frames=None):
        self.all_frames = all_frames        # list[str] base64 jpeg
        self.send_frames = send_frames      # list[str] actually sent to a VLM
        self.audio_path = audio_path         # str | None
        self.duration = duration             # float seconds (0 if unknown)
        self.has_audio = has_audio           # bool
        self.tile = tile                     # "" for single frames, e.g. "3x3" if each image is a grid
        self.n_slow = n_slow                 # trailing hi-res keyframes in send_frames (slow stream)
        self.context_frames = context_frames or send_frames   # full (Kimi) set, adaptive-selected
        self.gemma_frames = gemma_frames or send_frames        # small (Gemma) 5-frame subset

    @property
    def n_frames(self) -> int:
        return len(self.all_frames)


async def _run(cmd: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, out, err


async def _download(url: str, dest: str, client: httpx.AsyncClient, timeout: float) -> None:
    async with client.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in r.aiter_bytes(1 << 16):
                f.write(chunk)


async def _probe(path: str) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio_stream). Best-effort; (0, False) on failure."""
    try:
        rc, out, _ = await _run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path], timeout=20)
        if rc != 0:
            return 0.0, False
        info = json.loads(out.decode("utf-8", "ignore"))
        dur = 0.0
        try:
            dur = float(info.get("format", {}).get("duration", 0.0))
        except (TypeError, ValueError):
            dur = 0.0
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        return dur, has_audio
    except Exception as e:
        log.warning("ffprobe failed: %s", e)
        return 0.0, False


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _even_subset(items: list, k: int) -> list:
    if k <= 0 or len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _small_gray(path: str, size: int = 32) -> bytes:
    with Image.open(path) as im:
        return im.convert("L").resize((size, size)).tobytes()


def _frame_diff(a: bytes, b: bytes) -> float:
    """Mean absolute pixel difference of two equal-size gray thumbnails, normalized 0-1."""
    if not a or not b or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / (len(a) * 255.0)


def _adaptive_select(paths: list, window: int, threshold: float, max_frames: int) -> list:
    """Keep-on-change selector: walk frames, keep one when it differs enough from the LAST KEPT
    frame; force-keep every `window` frames so we always progress. Dedups static stretches,
    never drops a real transition (slow drift accumulates vs the reference and gets caught)."""
    if not paths:
        return []
    try:
        grays = [_small_gray(p) for p in paths]
    except Exception as e:
        log.warning("adaptive gray failed (%s); even sampling instead", e)
        return _even_subset(paths, max_frames)
    kept = [0]
    ref = grays[0]
    since = 0
    for i in range(1, len(paths)):
        since += 1
        if _frame_diff(ref, grays[i]) >= threshold or since >= window:
            kept.append(i)
            ref = grays[i]
            since = 0
            if len(kept) >= max_frames:
                break
    return [paths[i] for i in kept]


async def fetch_and_extract(task_id: str, video_url: str, client: httpx.AsyncClient,
                            per_clip_budget: float) -> Media:
    work = os.path.join(Config.WORK_DIR, task_id)
    os.makedirs(work, exist_ok=True)
    frames_dir = os.path.join(work, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    src = os.path.join(work, "src")
    audio = os.path.join(work, "audio.wav")

    # Give the network ~55% of the per-clip budget, leaving time for models.
    dl_timeout = max(15.0, min(60.0, per_clip_budget * 0.55))
    await _download(video_url, src, client, dl_timeout)

    duration, has_audio = await _probe(src)
    ff_timeout = max(20.0, per_clip_budget * 0.5)
    tile = Config.FRAME_TILE.strip().lower()

    if tile and "x" in tile:
        # Grid montage: pack many frames into a few tiled images so one image carries
        # e.g. 9 sequential frames — 4 grids ≈ 36 frames within a 5-image provider cap.
        cols, rows = (int(x) for x in tile.split("x", 1))
        per_grid = max(1, cols * rows)
        grids = max(1, Config.GRID_TOTAL_FRAMES // per_grid)
        dur = duration if duration and duration > 0 else grids * per_grid
        fps = max(0.1, (grids * per_grid) / dur)
        px = Config.GRID_FRAME_PX
        vf = (f"fps={fps:.4f},scale={px}:{px}:force_original_aspect_ratio=decrease"
              f":force_divisible_by=2,tile={cols}x{rows}")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
               "-vf", vf, "-frames:v", str(grids), "-q:v", "4",
               os.path.join(frames_dir, "f_%04d.jpg")]
    else:
        if duration and duration > 0:
            fps = min(Config.FRAME_FPS_MAX, Config.MAX_FRAMES / duration)
        else:
            fps = Config.FRAME_FPS_MAX
        fps = max(fps, 0.1)
        vf = (f"fps={fps:.4f},scale={Config.FRAME_PX}:{Config.FRAME_PX}"
              f":force_original_aspect_ratio=decrease")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
               "-vf", vf, "-frames:v", str(Config.MAX_FRAMES), "-q:v", "3",
               os.path.join(frames_dir, "f_%04d.jpg")]
    if has_audio:
        cmd += ["-map", "0:a?", "-ac", "1", "-ar", "16000", "-vn", audio]

    rc, _, err = await _run(cmd, timeout=ff_timeout)
    if rc != 0:
        log.warning("[%s] ffmpeg rc=%s: %s", task_id, rc, err.decode("utf-8", "ignore")[:200])

    frame_files = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    is_grid = bool(tile and "x" in tile)
    # CONTEXT set (Kimi, no cap): adaptive keep-on-change selection dedups static stretches
    # and keeps transitions, capped at KIMI_MAX_FRAMES.
    if is_grid:
        context_paths = frame_files
    elif Config.ADAPTIVE_FRAMES:
        context_paths = _adaptive_select(frame_files, Config.SCENE_WINDOW,
                                         Config.SCENE_DIFF_THRESHOLD, Config.KIMI_MAX_FRAMES)
    else:
        context_paths = _even_subset(frame_files, Config.KIMI_MAX_FRAMES)
    context_frames = [_b64(p) for p in context_paths]
    all_frames = context_frames
    send_frames = context_frames if is_grid else _even_subset(context_frames, Config.FRAMES_PER_REQUEST)
    # GEMMA set: a small representative subset of the KEPT frames (its provider cap).
    gemma_frames = context_frames if is_grid else _even_subset(context_frames, Config.GEMMA_FRAMES)
    audio_ok = has_audio and os.path.exists(audio) and os.path.getsize(audio) > 1024

    # Slow stream (SlowFast-LLaVA): a few full-resolution keyframes appended after the
    # grids, for fine detail the low-res tiles can't carry (text, colors, counts).
    n_slow = 0
    if Config.SLOW_FRAMES > 0 and duration and duration > 0:
        for k in range(Config.SLOW_FRAMES):
            ts = duration * (k + 1) / (Config.SLOW_FRAMES + 1)
            slow_path = os.path.join(work, f"slow_{k}.jpg")
            rc2, _, _ = await _run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{ts:.2f}", "-i", src, "-frames:v", "1",
                 "-vf", f"scale={Config.SLOW_PX}:{Config.SLOW_PX}:force_original_aspect_ratio=decrease",
                 "-q:v", "3", slow_path], timeout=30)
            if rc2 == 0 and os.path.exists(slow_path) and os.path.getsize(slow_path) > 1024:
                send_frames = send_frames + [_b64(slow_path)]
                n_slow += 1

    # We no longer need the source video or frame files on disk (N-03).
    try:
        os.remove(src)
    except OSError:
        pass

    if not all_frames:
        raise RuntimeError(f"[{task_id}] no frames extracted")

    log.info("[%s] dur=%.1fs pool=%d context=%d gemma=%d adaptive=%s audio=%s",
             task_id, duration, len(frame_files), len(context_frames), len(gemma_frames),
             bool(Config.ADAPTIVE_FRAMES), audio_ok)
    return Media(all_frames, send_frames, audio if audio_ok else None, duration, audio_ok,
                 tile if is_grid else "", n_slow,
                 context_frames=context_frames, gemma_frames=gemma_frames)


def cleanup(task_id: str) -> None:
    shutil.rmtree(os.path.join(Config.WORK_DIR, task_id), ignore_errors=True)
