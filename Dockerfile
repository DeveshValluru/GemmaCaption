# bestofn = v16g2 (the CONFIRMED 0.87 recipe) with best-of-N + description-based selection for the
# WITTY styles only. GENERATION is byte-for-byte v16g2 (same one-shot prompt, same personas, same
# exemplars, same temps) — the witty styles are just drawn N times, then a grounded scene description
# (source of truth) selects the best per style (accuracy gate -> wit pick). Formal is untouched single-shot.
# NO json mode, NO distinctness guard (both proven to crater to 0.33). First candidate == v16g2 draft,
# so selection can only swap in a peer; worst case falls back to plain one-shot.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORK_DIR=/tmp/stylecap

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

# --- v16g2 baked config (0.87) + MODE=bestofn ---
ENV MODE=bestofn \
    BESTOFN=4 \
    PERCEPTION_BACKEND=hf \
    STYLIZE_BACKEND=hf \
    FALLBACK_CHAIN=hf,fireworks,template \
    HF_BASE_URL=https://router.huggingface.co/v1 \
    HF_MODEL=google/gemma-4-31B-it:fastest \
    HF_MAX_IMAGES=5 \
    FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
    FIREWORKS_VLM_MODEL=accounts/fireworks/models/kimi-k2p6 \
    FIREWORKS_TEXT_MODEL=accounts/fireworks/models/kimi-k2p6 \
    FRAME_TILE= \
    FRAMES_PER_REQUEST=16 \
    FRAME_PX=1024 \
    MAX_FRAMES=72 \
    N_CANDIDATES=1 \
    USE_AUDIO=0 \
    ADAPTIVE_FRAMES=0 \
    FORMAL_TEMP=0.35 \
    WITTY_TEMP=0.9 \
    REASONING_EFFORT=none \
    HTTP_TIMEOUT_S=150 \
    CONCURRENCY=3 \
    PER_CLIP_TIMEOUT_S=180 \
    HTTP_RETRIES=6

ARG FIREWORKS_API_KEY=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

ENTRYPOINT ["python", "-m", "app.main"]
