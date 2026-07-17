# StyleCap — Build & Run Instructions (AMD Hackathon Track 2: Video Captioning)

This container reads a list of videos, generates **4 styled captions** for each
(`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) using Gemma, and writes
the results. Below is everything needed to build and test it **with your own API keys**
(none are included here).

---

## 1. What you need

- **Docker** (Docker Desktop is fine).
- The project files: the `app/` folder, `requirements.txt`, and `Dockerfile.bestofn`.
  (You do **not** need — and should not have — any `.env` file or baked keys.)
- **Your own API keys** (get your own; do not reuse anyone else's):
  - **`HF_TOKEN`** — *required.* A Hugging Face access token.
    - Create at <https://huggingface.co/settings/tokens>.
    - It must have **Inference Providers** access **and** some inference credit/quota.
    - You must **accept the Gemma license** on the model page first, or the token can't
      use it: <https://huggingface.co/google/gemma-4-31B-it> (click "Agree/Access").
  - **`FIREWORKS_API_KEY`** — *optional but recommended.* A fallback perceiver so a clip
    still gets a real caption if Gemma hiccups. Get one at <https://fireworks.ai>.
    Without it, a failed Gemma call falls back to a generic template caption.

---

## 2. The input/output contract (how the judge talks to the container)

The container reads **`/input/tasks.json`** and writes **`/output/results.json`**.
Those exact paths are mounted in by the harness — the code already points at them.

**Input — `/input/tasks.json`** is a JSON array of tasks. Each task has a `task_id`, a
`video_url` (the container downloads it itself), and the list of requested `styles`:

```json
[
  {
    "task_id": "clip-001",
    "video_url": "https://example.com/some-clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

**Output — `/output/results.json`** is a JSON array with one entry per task, containing a
caption for **every** requested style:

```json
[
  {
    "task_id": "clip-001",
    "captions": {
      "formal": "A woman sits at a white desk typing at a monitor ...",
      "sarcastic": "A truly riveting display of ...",
      "humorous_tech": "She debugs a production incident live ...",
      "humorous_non_tech": "She stares at her desk with pure betrayal ..."
    }
  }
]
```

---

## 3. Build the image (bake YOUR keys at build time)

Keys are passed as `--build-arg` — they are **not** written into the Dockerfile.
Run this from the project root (single line so it works in any shell):

```bash
docker build -f Dockerfile.bestofn --platform linux/amd64 --build-arg HF_TOKEN=hf_XXXXXXXX --build-arg FIREWORKS_API_KEY=fw_XXXXXXXX -t stylecap:mine .
```

- Replace `hf_XXXXXXXX` / `fw_XXXXXXXX` with your keys. If you skip Fireworks, just omit
  that `--build-arg`.
- **`--platform linux/amd64` is important** — the judge runs amd64. If you're on an Apple
  Silicon Mac and skip it, you'll build an arm64 image the judge can't run.

---

## 4. Test it exactly how the judge runs it (no env vars injected)

The judge runs the container with **only two volume mounts and nothing else** — no
environment variables, no keys (yours are already baked in). Reproduce that:

```bash
mkdir -p test/input test/output
# put a tasks.json (see section 2) into test/input/
docker run --rm -v "$(pwd)/test/input:/input:ro" -v "$(pwd)/test/output:/output" stylecap:mine
cat test/output/results.json
```

- On Windows PowerShell, use `${PWD}` instead of `$(pwd)`, or absolute paths.
- **Do NOT pass `--env-file` or `-e` when testing.** An env file can override the baked
  config and silently route to the wrong model — then you're testing the wrong pipeline.

**Sanity check the startup log:** it should read
`MODE=bestofn perception=hf ... chain=['hf','fireworks','template']`
and each clip should log **`done via bestofn:hf`**. If it says `done via fireworks` or
`done via template`, Gemma isn't being used — check that the HF token is valid, has credit,
and has accepted the Gemma license.

---

## 5. Rules the container must satisfy

- **Platform:** linux/amd64.
- **I/O paths:** read `/input/tasks.json`, write `/output/results.json` (already wired up).
- **Self-contained keys:** the judge injects **no** environment variables, so keys must be
  **baked at build time** (section 3). Don't rely on runtime `-e`/`--env-file`.
- **Completeness:** every requested style for every task must get a non-empty caption. The
  pipeline guarantees this — it falls back to a template caption if every model fails — so
  `results.json` is always valid and complete.
- **Time budget:** the container has an internal watchdog (~510s) and must finish under the
  harness cap (~600s). It downloads videos and calls the model over the network, so the
  machine needs internet access.
- **Entry point:** `python -m app.main` (already set in the Dockerfile — don't change it).

---

## 6. Security (important)

- **Never share or commit `.env`, tokens, or any file containing keys.** Each person bakes
  their own keys with `--build-arg`.
- If you push the image to a registry, remember the keys are **inside** the image — keep it
  **private**, or rotate the keys afterward.
