"""Runtime configuration (BLUEPRINT F-11).

All settings come from env vars WITH working baked defaults, because the Track 2
harness injects nothing (R-14). Nothing here raises on missing values; a missing
credential simply makes that backend unavailable and the pipeline falls to the
next tier (F-05).
"""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (ValueError, TypeError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (ValueError, TypeError):
        return default


def _str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


class Config:
    # ---- mode / backends ----
    MODE = _str("MODE", "oneshot")                    # oneshot | twostage
    PERCEPTION_BACKEND = _str("PERCEPTION_BACKEND", "fireworks")
    STYLIZE_BACKEND = _str("STYLIZE_BACKEND", "fireworks")

    # Ordered fallback chain for any model call (F-05). Names must resolve in
    # backends.build_registry(). template is always last and never fails.
    FALLBACK_CHAIN = [
        b.strip() for b in _str("FALLBACK_CHAIN", "fireworks,gemma,gemini,template").split(",") if b.strip()
    ]

    # ---- Tier 1: Gemma 4 on MI300X (vLLM, OpenAI-compatible) ----
    GEMMA_BASE_URL = _str("GEMMA_BASE_URL")
    GEMMA_API_KEY = _str("GEMMA_API_KEY")
    GEMMA_MODEL = _str("GEMMA_MODEL", "google/gemma-4-12B-it")

    # ---- Tier 2: Fireworks serverless ----
    FIREWORKS_API_KEY = _str("FIREWORKS_API_KEY")
    FIREWORKS_BASE_URL = _str("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    FIREWORKS_VLM_MODEL = _str("FIREWORKS_VLM_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct")
    FIREWORKS_TEXT_MODEL = _str("FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct")

    # ---- Tier 3: Gemini (OpenAI-compat surface) ----
    GEMINI_API_KEY = _str("GEMINI_API_KEY")
    GEMINI_BASE_URL = _str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODEL = _str("GEMINI_MODEL", "gemini-2.5-flash")

    # ---- Claude via Anthropic API (OpenAI-compat surface) — the strong perceiver (ClipTone's edge) ----
    ANTHROPIC_API_KEY = _str("ANTHROPIC_API_KEY")
    ANTHROPIC_BASE_URL = _str("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    ANTHROPIC_MODEL = _str("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # ---- Gemma 4 via HF Inference Providers (serverless, Gemma-prize path) ----
    HF_TOKEN = _str("HF_TOKEN")
    HF_BASE_URL = _str("HF_BASE_URL", "https://router.huggingface.co/v1")
    HF_MODEL = _str("HF_MODEL", "google/gemma-4-31B-it:fastest")
    # Second HF backend as fallback tier (bake-off runner-up; used only if Gemma fails)
    HF2_MODEL = _str("HF2_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:fastest")
    HF_MAX_IMAGES = _int("HF_MAX_IMAGES", 5)   # HF Gemma provider caps images/request at 5
    USE_AUDIO = _int("USE_AUDIO", 1)           # transcribe audio (Whisper via HF) into the fact-sheet
    HF_ASR_MODEL = _str("HF_ASR_MODEL", "openai/whisper-large-v3-turbo")

    # ---- frame grids (pack many frames into few images, beating the 5-image cap) ----
    FRAME_TILE = _str("FRAME_TILE", "")        # e.g. "3x3" enables grid montage; "" = single frames
    GRID_TOTAL_FRAMES = _int("GRID_TOTAL_FRAMES", 36)   # total frames sampled across the whole clip
    GRID_FRAME_PX = _int("GRID_FRAME_PX", 384)          # per-frame size inside the grid

    # ---- SlowFast two-stream input (SlowFast-LLaVA 2407.15841): grids = fast/motion
    # stream; a few full-resolution keyframes = slow/detail stream (OCR, colors, counts).
    SLOW_FRAMES = _int("SLOW_FRAMES", 0)       # hi-res keyframes appended after the grids
    SLOW_PX = _int("SLOW_PX", 768)

    # ---- pipeline params ----
    N_CANDIDATES = _int("N_CANDIDATES", 8)
    CONCURRENCY = _int("CONCURRENCY", 5)
    PER_CLIP_TIMEOUT_S = _int("PER_CLIP_TIMEOUT_S", 75)     # F-04
    WATCHDOG_S = _int("WATCHDOG_S", 510)                   # F-07 (< 600s cap, R-04)
    HTTP_TIMEOUT_S = _int("HTTP_TIMEOUT_S", 30)            # I-01
    HTTP_RETRIES = _int("HTTP_RETRIES", 2)                 # F-04
    # Reasoning models (e.g. Kimi K2.6) dump chain-of-thought into the response and
    # blow the token budget before emitting the answer. "none" disables it (fast + parseable).
    # Set empty for backends that reject the param.
    REASONING_EFFORT = _str("REASONING_EFFORT", "none")
    MAX_FRAMES = _int("MAX_FRAMES", 60)
    FRAME_PX = _int("FRAME_PX", 512)
    # ---- adaptive (content-aware) frame selection: dedup static stretches, keep transitions ----
    ADAPTIVE_FRAMES = _int("ADAPTIVE_FRAMES", 0)          # 1 = enable the keep-on-change selector
    SCENE_WINDOW = _int("SCENE_WINDOW", 8)                # force-keep a frame every N even if static
    SCENE_DIFF_THRESHOLD = _float("SCENE_DIFF_THRESHOLD", 0.10)  # mean 32x32 gray diff to count as "changed"
    KIMI_MAX_FRAMES = _int("KIMI_MAX_FRAMES", 24)         # cap on the full (Kimi) context frame set
    GEMMA_FRAMES = _int("GEMMA_FRAMES", 5)               # frames Gemma actually sees (its provider cap)
    FRAME_FPS_MAX = _float("FRAME_FPS_MAX", 1.0)
    FRAMES_PER_REQUEST = _int("FRAMES_PER_REQUEST", 16)    # how many frames we actually send to a VLM
    TWO_STAGE_VERIFY = _int("TWO_STAGE_VERIFY", 1)         # perceive self-check pass (twostage mode)
    DEBUG_CONTEXT = _int("DEBUG_CONTEXT", 0)               # write Pass-1 context into results.json (local debug)
    VERIFY_PASS = _int("VERIFY_PASS", 0)                  # v29: re-watch the clip, strip unsupported/sign/brand/identity claims before styling
    FORMAL_TEMP = _float("FORMAL_TEMP", 0.35)             # low temp for factual styles (precision)
    WITTY_TEMP = _float("WITTY_TEMP", 0.9)                # high temp for sarcastic/humour styles (wit knob; >0.9 risks garble)
    WORD_MIN = _int("WORD_MIN", 8)                        # caption length band — task wants "a caption", not a description
    WORD_MAX = _int("WORD_MAX", 32)                       # 8-32w: accuracy+style_match are the only axes; length is never rewarded
    JSON_MODE = _int("JSON_MODE", 0)                      # force response_format json_object (robustness, fewer parse-fails)
    DISTINCT_GUARD = _int("DISTINCT_GUARD", 0)            # regenerate a style whose caption overlaps another too much (style-match)
    DISTINCT_THRESHOLD = _float("DISTINCT_THRESHOLD", 0.6)  # word-overlap ratio above which two captions count as too similar
    # Which styles include register exemplars in their prompt block. Exemplars anchor register but may
    # LIMIT the witty styles (their humor echoes the samples' structures) — set e.g. "formal" to A/B that.
    EXEMPLAR_STYLES = [s.strip() for s in _str(
        "EXEMPLAR_STYLES", "formal,sarcastic,humorous_tech,humorous_non_tech").split(",") if s.strip()]
    # best-of-N for the witty styles (MODE=bestofn): draw N candidates, then a grounded scene
    # description drives selection. 1 = off (== v16g2). Generation itself is untouched v16g2.
    BESTOFN = _int("BESTOFN", 4)

    # ---- I/O ----
    INPUT_PATH = _str("INPUT_PATH", "/input/tasks.json")
    OUTPUT_PATH = _str("OUTPUT_PATH", "/output/results.json")
    WORK_DIR = _str("WORK_DIR", "/tmp/stylecap")

    @classmethod
    def summary(cls) -> str:
        """Non-secret snapshot for startup logging (never prints keys)."""
        def has(v: str) -> str:
            return "set" if v else "MISSING"
        return (
            f"MODE={cls.MODE} perception={cls.PERCEPTION_BACKEND} stylize={cls.STYLIZE_BACKEND} "
            f"chain={cls.FALLBACK_CHAIN} | keys: fireworks={has(cls.FIREWORKS_API_KEY)} "
            f"gemma_url={has(cls.GEMMA_BASE_URL)} gemini={has(cls.GEMINI_API_KEY)} | "
            f"N={cls.N_CANDIDATES} conc={cls.CONCURRENCY} clip_to={cls.PER_CLIP_TIMEOUT_S}s "
            f"watchdog={cls.WATCHDOG_S}s frames={cls.MAX_FRAMES}@{cls.FRAME_PX}px send={cls.FRAMES_PER_REQUEST}"
        )
