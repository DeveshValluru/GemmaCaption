"""Captioning pipeline (BLUEPRINT F-02, F-05, R-10, R-13).

Two modes, selected by MODE:
- oneshot  : one vision call returns all 4 styles (fast, simple; v1 baseline).
- twostage : perceive -> self-verify -> stylize each style with cross-style awareness
             (grounded facts written once, then rendered into each voice; v2).

Backends are tried in fallback order; if all fail, Tier-4 templates guarantee a
non-empty caption for every requested style.
"""
from __future__ import annotations

import asyncio
import logging
import re

from . import schema, styles
from .backends import ModelError, OpenAICompatBackend, ordered_backends, transcribe_audio
from .config import Config

log = logging.getLogger("stylecap.pipeline")


# ---------------------------------------------------------------- helpers
def _fill(caps: dict, style_list: list[str]) -> dict:
    """Guarantee every requested style is present and non-empty (R-10, F-02)."""
    for s in style_list:
        if not caps.get(s):
            caps[s] = styles.template_caption(s)
    return {s: caps[s] for s in style_list}


def _wc_ok(cap: str, lo: int = 6, hi: int = 130) -> bool:
    return lo <= len(cap.split()) <= hi


# more candidates for the harder-to-nail humor styles (best-of-N + select)
_CANDS = {"formal": 1, "sarcastic": 3, "humorous_tech": 5, "humorous_non_tech": 5}


def _n_candidates(style: str) -> int:
    return _CANDS.get(style, max(1, Config.N_CANDIDATES))


def _try_chain(designated, ordered):
    """Backend priority: the designated one first, then the rest of the chain."""
    seen = set()
    out = []
    for b in ([designated] if designated else []) + list(ordered):
        if b and id(b) not in seen:
            seen.add(id(b))
            out.append(b)
    return out


# ---------------------------------------------------------------- one-shot (v1)
def build_oneshot_prompt(style_list: list[str], n_frames: int) -> str:
    blocks = "\n\n".join(styles.style_block(s) for s in style_list)
    keys = ", ".join(f'"{s}"' for s in style_list)
    return (
        f"You are an expert video captioner. You are shown {n_frames} frames sampled evenly "
        "from a short video clip, in chronological order. Study them as a single moving scene.\n\n"
        "Write ONE caption for EACH requested style below.\n\n"
        "HARD RULES:\n"
        "- Be DETAILED but TIGHT: aim for about 90-120 words per caption (4-6 sentences). Describe "
        "the main subject(s) and attributes, the actions in order, the setting, and the most notable "
        "true specifics (on-screen text, colors, objects, counts). Enough real detail to be accurate, "
        "short enough to keep the style sharp — do NOT ramble past ~120 words.\n"
        "- Ground EVERY detail ONLY in what is visibly present across the frames. Never invent "
        "names, brands, exact locations, dates, or counts you cannot actually see — a wrong detail "
        "hurts your score more than a missing one.\n"
        "- Write the WHOLE caption in the requested style's voice: the style colors the entire "
        "description from start to finish, never just one clause. English only.\n"
        "- No preamble, no markdown, no style labels inside the captions.\n"
        "- Every caption describes the SAME scene, each in its own unmistakable style.\n\n"
        f"{blocks}\n\n"
        f"Return ONLY a JSON object with exactly these keys: {keys}\n"
        'Example shape: {"formal": "...", "sarcastic": "..."}'
    )


async def _retry_missing(backend, frames, missing, n_frames):
    prompt = build_oneshot_prompt(missing, n_frames)
    raw = await backend.chat_vision(frames, prompt, temperature=0.6, max_tokens=900, n=1)
    return schema.coerce_captions(schema.extract_json(raw), missing)


async def _guard_nontech(backend, frames, caps, n_frames):
    """Regenerate humorous_non_tech if it leaked banned tech vocabulary (style-score guard)."""
    cap = caps.get("humorous_non_tech", "")
    if not cap or not any(w in cap.lower().split() for w in styles.NON_TECH_BANNED):
        return caps
    prompt = build_oneshot_prompt(["humorous_non_tech"], n_frames) + (
        "\n\nIMPORTANT: Use ONLY everyday, non-technical words. Do NOT use screen, computer, "
        "phone, app, or ANY technology term. Write it like a funny everyday moment."
    )
    try:
        raw = await backend.chat_vision(frames, prompt, temperature=0.9, max_tokens=400, n=1)
        c = schema.coerce_captions(schema.extract_json(raw), ["humorous_non_tech"]).get("humorous_non_tech", "")
        if c and not any(w in c.lower().split() for w in styles.NON_TECH_BANNED):
            caps["humorous_non_tech"] = c
    except ModelError:
        pass
    return caps


# Per-style temperature: factual styles want precision (low temp), witty styles want creative leaps
# (high temp). Splitting the styles into two temperature groups is the "wit knob".
_WITTY_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}


def _temp_groups(style_list: list[str]) -> list[tuple[list[str], float]]:
    plain = [s for s in style_list if s not in _WITTY_STYLES]
    witty = [s for s in style_list if s in _WITTY_STYLES]
    groups = []
    if plain:
        groups.append((plain, Config.FORMAL_TEMP))
    if witty:
        groups.append((witty, Config.WITTY_TEMP))
    return groups


# Foreign-script ranges (Cyrillic, Arabic, Devanagari, Tamil, CJK, Kana, Hangul) — never valid in an
# English caption; their presence signals a high-temperature coherence breakdown.
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ஀-௿一-鿿぀-ヿ가-힯]")


def _looks_garbled(text: str) -> bool:
    return bool(text) and bool(_FOREIGN_SCRIPT.search(text))


async def _gen_by_temp(backend, frames, style_list, prompt_fn, max_tokens=1200):
    """Generate captions grouping styles by temperature (formal cold, witty hot). Partial-success safe."""
    caps: dict = {}
    for grp, temp in _temp_groups(style_list):
        try:
            raw = await backend.chat_vision(frames, prompt_fn(grp), temperature=temp, max_tokens=max_tokens, n=1)
            caps.update(schema.coerce_captions(schema.extract_json(raw or ""), grp))
        except ModelError as e:
            log.info("temp-group %s failed on %s: %s", grp, backend.name, e)
    # coherence guard: a hot-temp caption with foreign-script garble is regenerated once at a safe low temp
    for s in [s for s in style_list if _looks_garbled(caps.get(s, ""))]:
        try:
            raw = await backend.chat_vision(frames, prompt_fn([s]), temperature=0.5, max_tokens=max_tokens, n=1)
            fixed = schema.coerce_captions(schema.extract_json(raw or ""), [s]).get(s, "")
            if fixed and not _looks_garbled(fixed):
                caps[s] = fixed
        except ModelError:
            pass
    return caps


async def _distinct_guard(backend, frames, caps, style_list, n_frames):
    """Regenerate any style whose caption overlaps another too much — guarantees 4 distinct voices
    (blurred styles lose on the style-match axis). Post-generation check, never touches the base prompt."""
    def _overlap(a, b):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0
    present = [s for s in style_list if caps.get(s)]
    to_fix: list[str] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            if _overlap(caps[present[i]], caps[present[j]]) > Config.DISTINCT_THRESHOLD and present[j] not in to_fix:
                to_fix.append(present[j])   # regenerate the later of a too-similar pair
    for s in to_fix:
        try:
            prompt = build_oneshot_prompt([s], n_frames) + (
                "\n\nIMPORTANT: word this caption so it is CLEARLY different in structure, imagery, and "
                "phrasing from the other styles — no shared sentences or repeated jokes.")
            temp = Config.WITTY_TEMP if s in _WITTY_STYLES else 0.6
            c = schema.coerce_captions(
                schema.extract_json(await backend.chat_vision(frames, prompt, temperature=temp,
                                                              max_tokens=1000, n=1) or ""), [s]).get(s, "")
            if c:
                caps[s] = c
        except ModelError:
            pass
    return caps


async def caption_oneshot(task: dict, media, backends: list[OpenAICompatBackend]):
    style_list = task["styles"]
    frames = media.send_frames
    n = len(frames)

    last_err = None
    for b in backends:
        try:
            caps = await _gen_by_temp(b, frames, style_list, lambda grp: build_oneshot_prompt(grp, n), max_tokens=1200)
            missing = [s for s in style_list if not caps.get(s)]
            if missing:
                try:
                    caps.update(await _retry_missing(b, frames, missing, n))
                except ModelError as e:
                    log.info("[%s] retry_missing on %s failed: %s", task["task_id"], b.name, e)
            if "humorous_non_tech" in style_list:
                caps = await _guard_nontech(b, frames, caps, n)
            if Config.DISTINCT_GUARD:
                caps = await _distinct_guard(b, frames, caps, style_list, n)
            if caps:
                return _fill(caps, style_list), b.name
        except ModelError as e:
            last_err = e
            log.warning("[%s] backend %s failed: %s", task["task_id"], b.name, e)
            continue
    log.warning("[%s] all model backends failed (%s) -> templates", task["task_id"], last_err)
    return _fill({}, style_list), "template"


# ---------------------------------------------------------------- best-of-N witty + description-select (v16g2bon)
# Generation = the winning v16g2 one-shot (Gemma sees frames), UNCHANGED. The witty styles are drawn N times
# at high temp; a SEPARATE grounded scene description (frames -> facts, all-Gemma so prize-safe) is the
# selection-only source of truth. Two-criterion pick: (1) drop candidates that contradict the description
# (accuracy), (2) choose the wittiest survivor with an anti-bland bias. cand_sets[0] == the v16g2 draft, so
# a failed/degenerate selection can only fall back to the v16g2 caption, never to junk.
def _bon_select_prompt(description: str, style: str, cands: list[str]) -> str:
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (
        "Below is a FACTUAL description of a short video (the ground truth), then several candidate "
        f"captions written in the '{style}' style.\n\n"
        f"--- VIDEO FACTS ---\n{description}\n--- END FACTS ---\n\n"
        f"CANDIDATES:\n{listing}\n\n"
        f"Target style: {style} — {styles.style_def(style)}\n\n"
        "Choose the SINGLE best caption in two steps:\n"
        "1. ACCURACY GATE — disqualify any candidate that asserts something NOT supported by the video "
        "facts (an invented object, number, name, brand, or action). One wrong detail disqualifies it.\n"
        "2. STYLE PICK — among the survivors, take the one that best nails the style: the boldest, "
        "funniest, most specific line about THIS video. Reward sharp, on-target wit; reject the bland, "
        "generic, or safe-but-boring option. Do NOT prefer a caption merely for being longer.\n"
        "Reply with ONLY the number of the best caption."
    )


async def _bon_select(backend, description, style, cands):
    cands = [c for c in cands if c]
    if len(cands) <= 1:
        return cands[0] if cands else ""
    if not description:
        return cands[0]                      # no ground truth -> keep the v16g2 first draft
    try:
        r = await backend.chat_text(_bon_select_prompt(description, style, cands),
                                    temperature=0.0, max_tokens=8, n=1)
    except ModelError:
        return cands[0]
    m = re.search(r"\d+", r or "")
    if m:
        i = int(m.group()) - 1
        if 0 <= i < len(cands):
            return cands[i]
    return cands[0]


async def caption_bestofn(task: dict, media, reg, ordered):
    style_list = task["styles"]
    frames = media.send_frames
    n = len(frames)
    witty = [s for s in style_list if s in _WITTY_STYLES]
    plain = [s for s in style_list if s not in _WITTY_STYLES]
    for b in ordered:
        try:
            caps: dict = {}
            # 1. plain styles (formal): single shot, identical to v16g2
            if plain:
                raw = await b.chat_vision(frames, build_oneshot_prompt(plain, n),
                                          temperature=Config.FORMAL_TEMP, max_tokens=1200, n=1)
                caps.update(schema.coerce_captions(schema.extract_json(raw or ""), plain))
            # 2. witty styles: best-of-N at high temp (each draw yields ALL witty styles)
            cand_sets: list[dict] = []
            if witty:
                k = max(1, Config.BESTOFN)
                raws = await b.chat_vision(frames, build_oneshot_prompt(witty, n),
                                           temperature=Config.WITTY_TEMP, max_tokens=1200, n=k)
                for r in (raws if isinstance(raws, list) else [raws]):
                    d = schema.coerce_captions(schema.extract_json(r or ""), witty)
                    if any(d.get(s) for s in witty):
                        cand_sets.append(d)
            # 3. grounded description = selection source-of-truth (only worth it with >1 candidate set)
            desc = ""
            if witty and len(cand_sets) > 1:
                try:
                    desc = (await b.chat_vision(frames, _scene_prompt(n),
                                                temperature=0.2, max_tokens=1000, n=1) or "").strip()
                except ModelError:
                    desc = ""
            # 4. per-witty-style: accuracy-gate on desc, then wit-pick (garbled candidates dropped first)
            for s in witty:
                cands = [d.get(s, "") for d in cand_sets if d.get(s) and not _looks_garbled(d.get(s, ""))]
                if not cands:
                    cands = [d.get(s, "") for d in cand_sets if d.get(s)]
                caps[s] = await _bon_select(b, desc, s, cands)
            # 5. fills + guards (same safety net as v16g2)
            missing = [s for s in style_list if not caps.get(s)]
            if missing:
                try:
                    caps.update(await _retry_missing(b, frames, missing, n))
                except ModelError:
                    pass
            for s in [x for x in style_list if _looks_garbled(caps.get(x, ""))]:
                try:
                    raw = await b.chat_vision(frames, build_oneshot_prompt([s], n),
                                              temperature=0.5, max_tokens=1200, n=1)
                    fixed = schema.coerce_captions(schema.extract_json(raw or ""), [s]).get(s, "")
                    if fixed and not _looks_garbled(fixed):
                        caps[s] = fixed
                except ModelError:
                    pass
            if "humorous_non_tech" in style_list:
                caps = await _guard_nontech(b, frames, caps, n)
            if Config.DEBUG_CONTEXT:
                task["_context"] = {"evidence": desc}
                task["_pass1"] = {s: [d.get(s, "") for d in cand_sets] for s in witty}
            if any(caps.get(s) for s in style_list):
                return _fill(caps, style_list), f"bestofn:{b.name}"
        except ModelError as e:
            log.warning("[%s] bestofn %s failed: %s", task["task_id"], b.name, e)
            continue
    return await caption_oneshot(task, media, ordered)


# ---------------------------------------------------------------- two-stage (v2)
def _grid_note(tile: str, n_grids: int, n_slow: int = 0) -> str:
    if not tile:
        return ""
    note = (f"The first {n_grids} images are {tile} grids of sequential video frames "
            "(read left-to-right, then top-to-bottom = forward in time). Treat all the grids "
            "together as ONE continuous clip in chronological order.")
    if n_slow:
        note += (f" The last {n_slow} image(s) are FULL-RESOLUTION keyframes from the same clip — "
                 "use them to read fine detail reliably: on-screen text, colors, textures, counts, "
                 "faces, and small objects.")
    return note + "\n\n"


def _audio_note(transcript: str) -> str:
    if not transcript:
        return ""
    return (f'\n\nAUDIO TRANSCRIPT (words spoken or narrated in the clip): "{transcript}"\n'
            "Use this to inform what is happening (e.g. speech, narration, dialogue), but only "
            "state things actually shown in the frames or clearly said in the audio. If the "
            "transcript looks like background music, lyrics, or gibberish rather than real "
            "speech about the scene, IGNORE it entirely.")


def _perceive_prompt(n_grids: int, tile: str = "", transcript: str = "", n_slow: int = 0) -> str:
    unit = f"{n_grids} frame-grids" if tile else f"{n_grids} frames"
    return (
        _grid_note(tile, n_grids, n_slow) +
        f"You are analyzing {unit} sampled in order from a short video clip. "
        "Describe EXACTLY what happens, grounded only in what is visible.\n"
        "Cover: the main subject(s) and their attributes; the sequence of actions IN THE ORDER "
        "they actually occur (use 'first... then... finally' only for transitions you can "
        "actually see happening across the frames — never invent an action or transition); the "
        "setting/environment; camera movement; any on-screen text (verbatim, from the "
        "full-resolution keyframes if present); and any notable or unusual detail.\n"
        "Be specific and factual. Do NOT invent names, brands, exact places, dates, breeds, or "
        "counts you cannot actually see. Write 4-6 sentences of plain prose (no lists, no markdown)."
        + _audio_note(transcript)
    )


def _verify_prompt(draft: str, n_grids: int, tile: str = "", transcript: str = "",
                   n_slow: int = 0) -> str:
    # Claim-level verification (Kestrel/Woodpecker-style), folded into one call:
    # decompose -> check each claim against the images -> rewrite with survivors only.
    return (
        _grid_note(tile, n_grids, n_slow) +
        f"Here is a draft description of a video:\n\n{draft}\n\n"
        "Fact-check it claim by claim:\n"
        "1. Mentally list every factual claim the draft makes (subjects, attributes, actions, "
        "action ORDER, setting, camera, text, counts).\n"
        "2. Check each claim against the images. Mark it VERIFIED only if you can clearly see it; "
        "pay special attention to action order and transitions (a common error is describing a "
        "sequence that never happens) and to details only readable in the full-resolution "
        "keyframes (text, colors, counts). Be EXTRA strict about camera motion (pan vs zoom vs "
        "static — only claim what the frame sequence proves; if unsure, omit camera motion "
        "entirely) and about any detail visible in only a single keyframe (keep it only if "
        "unambiguous, e.g. clearly readable text).\n"
        "3. Rewrite the description keeping ONLY verified claims, fixing any wrong ones, and "
        "adding clearly-visible facts the draft missed. A shorter description with only certain "
        "facts beats a longer one with a single wrong claim.\n"
        "Return ONLY the rewritten description as plain prose (4-6 sentences)."
        + _audio_note(transcript)
    )


def _stylize_prompt(factsheet: str, style: str, prev: list[tuple[str, str]]) -> str:
    prev_txt = ""
    if prev:
        joined = "\n".join(f'- ({s}) "{c}"' for s, c in prev)
        prev_txt = ("\n\nCaptions already written for OTHER styles — do NOT reuse their wording, "
                    f"structure, or the same opening:\n{joined}")
    return (
        "Facts about a short video clip:\n\n"
        f"{factsheet}\n\n"
        "Write ONE detailed caption in the target style below.\n\n"
        "RULES:\n"
        "- Be thorough: 2-4 sentences describing the main subject, the actions in order, the "
        "setting, and notable specifics — a rich, complete description scores higher on accuracy.\n"
        "- Write the WHOLE caption in the style's voice, not just one clause; the humor/irony/tone "
        "must be ABOUT what is actually in this video, never a generic quip.\n"
        "- Use ONLY facts from the description above; invent nothing.\n\n"
        f"{styles.style_block(style)}{prev_txt}\n\n"
        "English only. No preamble, no markdown, no style label. Return ONLY the caption text."
    )


async def _perceive(backend, media, transcript: str = "") -> str:
    frames = media.send_frames
    n_slow = getattr(media, "n_slow", 0)
    n_grids = len(frames) - n_slow
    draft = (await backend.chat_vision(
        frames, _perceive_prompt(n_grids, media.tile, transcript, n_slow),
        temperature=0.2, max_tokens=500, n=1) or "").strip()
    if not draft:
        return ""
    if Config.TWO_STAGE_VERIFY:
        try:
            verified = (await backend.chat_vision(
                frames, _verify_prompt(draft, n_grids, media.tile, transcript, n_slow),
                temperature=0.2, max_tokens=600, n=1) or "").strip()
            if len(verified) >= 40:
                return verified
        except ModelError as e:
            log.info("verify step failed (%s); using draft", e)
    return draft


async def _judge_select(backend, factsheet, style, cands) -> str:
    """Judge-replica: pick the candidate best on accuracy x style (GRPO-at-inference)."""
    if len(cands) <= 1:
        return cands[0] if cands else ""
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    prompt = (
        f"Video facts (ground truth):\n{factsheet}\n\n"
        f"Requested style: {style} — {styles.style_def(style)}\n\n"
        f"Candidate captions:\n{listing}\n\n"
        "Pick the SINGLE best caption. It must (a) be accurate — every claim supported by the "
        "video facts, nothing invented; (b) clearly identify the main subject and action; and "
        "(c) nail the requested style — for humor/sarcasm, genuinely the wittiest and most "
        "on-target, not the blandest or the most generic. Prefer short and punchy over long and "
        "explainy. Reply with ONLY the number of the best caption."
    )
    try:
        r = await backend.chat_text(prompt, temperature=0.0, max_tokens=8, n=1)
    except ModelError:
        return cands[0]
    m = re.search(r"\d+", r or "")
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(cands):
            return cands[idx]
    return cands[0]


async def _stylize_one(backend, factsheet, style, prev, n: int = 1) -> str:
    prompt = _stylize_prompt(factsheet, style, prev)
    raw = await backend.chat_text(prompt, temperature=(0.95 if n > 1 else 0.8),
                                  max_tokens=320, n=n)
    cands = [schema.clean_caption(c) for c in (raw if isinstance(raw, list) else [raw])]
    cands = [c for c in cands if c and _wc_ok(c, 8, 130)]
    if not cands:  # last-ditch single plain attempt
        one = schema.clean_caption(await backend.chat_text(prompt, temperature=0.6,
                                                           max_tokens=320, n=1))
        return one
    # style guard: drop tech words from humorous_non_tech if any clean candidate survives
    if style == "humorous_non_tech":
        clean = [c for c in cands if not any(w in c.lower().split() for w in styles.NON_TECH_BANNED)]
        if clean:
            cands = clean
    return await _judge_select(backend, factsheet, style, cands)


async def caption_twostage(task: dict, media, reg, ordered):
    """Three specialist agents: perceive (best model, low volume) -> stylize (high volume,
    best-of-N per style) -> select. Perception and stylization can run on DIFFERENT backends
    (e.g. Kimi sees, Gemma/HF writes) so each job runs on the model that fits it."""
    style_list = task["styles"]
    perceive_chain = _try_chain(reg.get(Config.PERCEPTION_BACKEND), ordered)
    stylize_chain = _try_chain(reg.get(Config.STYLIZE_BACKEND), ordered)
    if not perceive_chain or not stylize_chain:
        return await caption_oneshot(task, media, ordered)

    transcript = ""
    if Config.USE_AUDIO and media.audio_path:
        transcript = await transcribe_audio(perceive_chain[0].client, media.audio_path)
        if transcript:
            log.info("[%s] audio: %s", task["task_id"], transcript[:80])

    # Stage 1 — PERCEIVE: rich, accurate description of the clip (Kimi first).
    factsheet, seen_by = "", ""
    for b in perceive_chain:
        try:
            factsheet = await _perceive(b, media, transcript)
            if factsheet:
                seen_by = b.name
                break
        except ModelError as e:
            log.warning("[%s] perceive %s failed: %s", task["task_id"], b.name, e)
    if not factsheet:
        log.warning("[%s] perceive failed -> oneshot fallback", task["task_id"])
        return await caption_oneshot(task, media, ordered)

    # Stage 2+3 — STYLIZE (best-of-N, more candidates for humor) + SELECT (HF, high volume).
    caps: dict = {}
    prev: list[tuple[str, str]] = []
    made_by = ""
    for st in style_list:
        k = _n_candidates(st)
        cap = ""
        for b in stylize_chain:
            try:
                cap = await _stylize_one(b, factsheet, st, prev, k)
                if cap:
                    made_by = b.name
                    break
            except ModelError:
                continue
        if cap:
            caps[st] = cap
            prev.append((st, cap))
    if not caps:
        log.warning("[%s] stylize produced nothing -> oneshot fallback", task["task_id"])
        return await caption_oneshot(task, media, ordered)
    return _fill(caps, style_list), f"{seen_by}+{made_by}"


# ---------------------------------------------------------------- two-pass grounded (v18+)
_GROUND_PREFILL = '{"evidence": "'
_REFINE_PREFILL = '{"formal": "'

_GROUND_SHAPE = (
    '{\n'
    '  "evidence": "<rich grounded description: the PRIMARY subject and what it does, actions in '
    'order, setting, colors, on-screen text verbatim or none, notable specifics>",\n'
    '  "subject": {"what": "<the ONE main subject>", "features": "<visible features confirming its '
    'identity>", "confidence": "high or low"},\n'
    '  "hooks": ["<2-3 real, grounded juxtapositions that are inherently funny or ironic>"],\n'
    '  "uncertain": ["<anything you are NOT sure of — must NOT be asserted in any caption>"],\n'
    '  "formal": "<caption>",\n'
    '  "sarcastic": "<caption>",\n'
    '  "humorous_tech": "<caption>",\n'
    '  "humorous_non_tech": "<caption>"\n'
    '}'
)


def _ground_prompt(style_list: list[str], n_frames: int, transcript: str = "") -> str:
    blocks = "\n\n".join(styles.style_block(s) for s in style_list)
    return (
        f"You are an expert video analyst. You see {n_frames} frames sampled in order from a short "
        "video clip. Produce ONE JSON object in two steps.\n\n"
        "STEP 1 — GROUND (use ONLY what is visibly present):\n"
        "- evidence: a rich, factual, DETAILED description. Grounded — never guess.\n"
        "- subject: identify the ONE main subject ONLY by visible features (ear/face shape, body "
        "proportions, tail, etc.). Cross-check it across at least 3 frames. If the frames are "
        "ambiguous or disagree, set confidence 'low'.\n"
        "- hooks: real, grounded juxtapositions that are inherently funny/ironic (so the humor styles "
        "never need to invent).\n"
        "- uncertain: anything not certain; it must NOT be asserted in captions."
        + _audio_note(transcript) +
        "\n\nSTEP 2 — STYLE (reframe ONLY the grounded facts into each caption):\n"
        "- Write a LONG, comprehensive caption of about 300-350 WORDS for EACH style. Describe the "
        "scene in exhaustive TRUE detail — every subject, action, attribute, setting element, color, "
        "text, and notable specific you can actually see — sustaining the style's voice throughout.\n"
        "- Humor/sarcasm come from the HOOKS (real juxtapositions), NEVER from invented facts, numbers, "
        "names, durations, or motives. Length must come from MORE TRUE detail, never from padding or "
        "speculation.\n"
        "- Write the WHOLE caption in the style's voice. If subject confidence is 'low', describe the "
        "subject GENERICALLY (e.g. 'a small animal') in every caption.\n\n"
        f"{blocks}\n\n"
        f"Return ONLY this JSON object:\n{_GROUND_SHAPE}"
    )


def _refine_prompt(data: dict, caps: dict, style_list: list[str], n_frames: int) -> str:
    draft = "\n".join(f'- {s}: "{caps.get(s, "")}"' for s in style_list)
    keys = ", ".join(f'"{s}"' for s in style_list)
    return (
        f"You see the SAME {n_frames} frames from the video. Here is a draft analysis and captions:\n\n"
        f"EVIDENCE: {data.get('evidence', '')}\n"
        f"CLAIMED SUBJECT: {data.get('subject', {})}\n"
        f"MUST-NOT-ASSERT: {data.get('uncertain', [])}\n\n"
        f"DRAFT CAPTIONS:\n{draft}\n\n"
        "Refine every caption against the FRAMES. Rules:\n"
        "1. RE-VERIFY THE SUBJECT by its visible features. If the identification is wrong, fix it and "
        "every claim that depended on it.\n"
        "2. Remove any claim NOT supported by the frames (invented numbers, names, motives, actions); "
        "never assert anything on the must-not-assert list.\n"
        "3. KEEP the style, voice, and humor — only make every claim true. Keep specific TRUE details, "
        "add any obvious true detail the draft missed. Keep it a LONG ~300-350 word caption.\n"
        "4. Introduce NO new inventions.\n\n"
        f"Return ONLY JSON with the refined captions, keys: {keys}."
    )


async def _ground_and_style(backend, media, style_list, transcript=""):
    frames = media.send_frames
    raw = await backend.chat_vision(frames, _ground_prompt(style_list, len(frames), transcript),
                                    temperature=0.4, max_tokens=4000, n=1, prefill=_GROUND_PREFILL)
    data = schema.extract_json(_GROUND_PREFILL + (raw or "")) or schema.extract_json(raw or "") or {}
    return data, schema.coerce_captions(data, style_list)


async def _refine(backend, media, data, caps, style_list):
    frames = media.send_frames
    raw = await backend.chat_vision(frames, _refine_prompt(data, caps, style_list, len(frames)),
                                    temperature=0.3, max_tokens=2800, n=1, prefill=_REFINE_PREFILL)
    return schema.coerce_captions(
        schema.extract_json(_REFINE_PREFILL + (raw or "")) or schema.extract_json(raw or "") or {},
        style_list)


async def caption_twopass(task: dict, media, reg, ordered):
    """Pass 1: ground (evidence + subject-check + hooks) then style, one prefilled call.
    Pass 2: re-show frames, cut unsupported claims + re-verify subject. Pass 1 = guaranteed fallback."""
    style_list = task["styles"]
    frames = media.send_frames
    transcript = ""
    if Config.USE_AUDIO and media.audio_path and ordered:
        transcript = await transcribe_audio(ordered[0].client, media.audio_path)
        if transcript:
            log.info("[%s] audio: %s", task["task_id"], transcript[:80])
    for b in ordered:
        try:
            data, caps = await _ground_and_style(b, media, style_list, transcript)
            if not any(caps.get(s) for s in style_list):
                raise ModelError("empty pass-1")
            if Config.DEBUG_CONTEXT:  # stash Pass-1 context for the judge app
                task["_context"] = {k: data.get(k) for k in ("evidence", "subject", "hooks", "uncertain")}
                if transcript:
                    task["_context"]["audio"] = transcript
                task["_pass1"] = {s: caps.get(s, "") for s in style_list}
            base = _fill(dict(caps), style_list)          # Pass-1 = never-zero fallback
            try:
                refined = await _refine(b, media, data, base, style_list)
                caps = {s: (refined.get(s) or base.get(s)) for s in style_list}
            except ModelError as e:
                log.info("[%s] refine failed (%s); keeping pass-1", task["task_id"], e)
                caps = base
            if "humorous_non_tech" in style_list:
                caps = await _guard_nontech(b, frames, caps, len(frames))
            return _fill(caps, style_list), b.name
        except ModelError as e:
            log.warning("[%s] twopass %s failed: %s", task["task_id"], b.name, e)
            continue
    log.warning("[%s] twopass failed everywhere -> oneshot", task["task_id"])
    return await caption_oneshot(task, media, ordered)


# ---------------------------------------------------------------- kimigemma (v21)
# Kimi (full frames, no cap) builds CONTEXT  ‖  Gemma (5 frames) DRAFTS  ->  Gemma writes FINAL
# from draft + context (re-seeing its 5 frames to break conflicts). Gemma writes final = prize-eligible.
_CTX_PREFILL = '{"evidence": "'
_FINAL_PREFILL = '{"formal": "'


def _context_prompt(n_frames: int, transcript: str = "") -> str:
    return (
        f"You are an expert video analyst. You see {n_frames} frames spanning a short video clip "
        "(full coverage of the whole clip). Build a THOROUGH, strictly grounded CONTEXT — only what "
        "is actually visible. Identify the main subject ONLY by visible features and cross-check it "
        "across frames; set confidence 'low' if the frames are ambiguous. Never guess or invent."
        + _audio_note(transcript) +
        "\n\nReturn ONLY this JSON:\n"
        '{\n'
        '  "evidence": "<exhaustive grounded description: primary subject + what it does, the actions '
        'IN ORDER across the whole clip, setting, colors, on-screen text verbatim or none, notable specifics>",\n'
        '  "subject": {"what": "<main subject>", "features": "<visible features confirming identity>", "confidence": "high or low"},\n'
        '  "actions": ["<ordered actions across the clip>"],\n'
        '  "onscreen_text": "<verbatim or none>",\n'
        '  "hooks": ["<2-3 real, grounded funny/ironic juxtapositions>"],\n'
        '  "uncertain": ["<anything not certain — must NOT be asserted in captions>"]\n'
        '}'
    )


def _final_prompt(context: dict, draft: dict, style_list: list[str]) -> str:
    keys = ", ".join(f'"{s}"' for s in style_list)
    draft_txt = "\n".join(f'- {s}: "{draft.get(s, "")}"' for s in style_list)
    return (
        "You see 5 frames from a video. Below is a FULLER CONTEXT built from ALL frames of the clip "
        "(more coverage than your 5 frames), plus your own DRAFT captions.\n\n"
        f"CONTEXT (whole-clip analysis):\n"
        f"evidence: {context.get('evidence', '')}\n"
        f"subject: {context.get('subject', {})}\n"
        f"actions: {context.get('actions', [])}\n"
        f"on-screen text: {context.get('onscreen_text', '')}\n"
        f"hooks (use for humor): {context.get('hooks', [])}\n"
        f"MUST-NOT-ASSERT: {context.get('uncertain', [])}\n\n"
        f"YOUR DRAFT CAPTIONS:\n{draft_txt}\n\n"
        "Write the FINAL caption for each style:\n"
        "1. ENRICH each draft with TRUE details from the CONTEXT that your 5 frames missed (the "
        "context saw the whole clip).\n"
        "2. If the CONTEXT conflicts with your draft, TRUST the context but glance at your 5 frames "
        "to confirm; fix the subject and any wrong claim.\n"
        "3. NEVER assert anything on MUST-NOT-ASSERT. Humor comes from the hooks (real), never invented.\n"
        "4. Keep it about 90-120 words (4-6 sentences) in each style's unmistakable voice — detailed "
        "enough to be accurate, tight enough to keep the style sharp. Do NOT ramble past ~120 words.\n\n"
        f"Return ONLY JSON with keys: {keys}."
    )


async def _kimi_context(backend, media, transcript=""):
    frames = media.context_frames
    raw = await backend.chat_vision(frames, _context_prompt(len(frames), transcript),
                                    temperature=0.3, max_tokens=1600, n=1, prefill=_CTX_PREFILL)
    data = schema.extract_json(_CTX_PREFILL + (raw or "")) or schema.extract_json(raw or "")
    return data if isinstance(data, dict) else {}   # Kimi sometimes emits a JSON array; never .get() a list


async def _gemma_draft(backend, media, style_list):
    frames = media.gemma_frames
    prompt = build_oneshot_prompt(style_list, len(frames))
    raw = await backend.chat_vision(frames, prompt, temperature=0.5, max_tokens=2200, n=1)
    caps = schema.coerce_captions(schema.extract_json(raw or ""), style_list)
    missing = [s for s in style_list if not caps.get(s)]
    if missing:
        try:
            caps.update(await _retry_missing(backend, frames, missing, len(frames)))
        except ModelError:
            pass
    return caps


async def _gemma_final(backend, media, context, draft, style_list):
    frames = media.gemma_frames
    raw = await backend.chat_vision(frames, _final_prompt(context, draft, style_list),
                                    temperature=0.4, max_tokens=3200, n=1, prefill=_FINAL_PREFILL)
    return schema.coerce_captions(
        schema.extract_json(_FINAL_PREFILL + (raw or "")) or schema.extract_json(raw or "") or {},
        style_list)


async def caption_kimigemma(task: dict, media, reg, ordered):
    style_list = task["styles"]
    kimi = reg.get(Config.PERCEPTION_BACKEND)     # fireworks/Kimi — no image cap, full context set
    gemma = reg.get(Config.STYLIZE_BACKEND)       # hf/Gemma — 5-frame draft + final writer
    if not gemma:
        return await caption_oneshot(task, media, ordered)

    transcript = ""
    if Config.USE_AUDIO and media.audio_path:
        transcript = await transcribe_audio((kimi or gemma).client, media.audio_path)
        if transcript:
            log.info("[%s] audio: %s", task["task_id"], transcript[:80])

    async def _ctx():
        if not kimi:
            return {}
        try:
            return await _kimi_context(kimi, media, transcript)
        except ModelError as e:
            log.info("[%s] kimi context failed: %s", task["task_id"], e)
            return {}

    async def _draft():
        try:
            return await _gemma_draft(gemma, media, style_list)
        except ModelError as e:
            log.warning("[%s] gemma draft failed: %s", task["task_id"], e)
            return {}

    # Stage 1 (parallel): Kimi full-clip context  ‖  Gemma 5-frame draft
    context, draft = await asyncio.gather(_ctx(), _draft())
    if not any(draft.get(s) for s in style_list):
        return await caption_oneshot(task, media, ordered)
    base = _fill(dict(draft), style_list)

    # Stage 2: Gemma writes final from draft + context (re-sees its 5 frames)
    caps = base
    if context:
        try:
            final = await _gemma_final(gemma, media, context, base, style_list)
            caps = {s: (final.get(s) or base.get(s)) for s in style_list}
        except ModelError as e:
            log.info("[%s] final merge failed (%s); keeping draft", task["task_id"], e)

    if Config.DEBUG_CONTEXT:
        task["_context"] = {k: context.get(k) for k in ("evidence", "subject", "hooks", "uncertain")} if context else {}
        if transcript:
            task.setdefault("_context", {})["audio"] = transcript
        task["_pass1"] = {s: base.get(s, "") for s in style_list}

    if "humorous_non_tech" in style_list:
        caps = await _guard_nontech(gemma, media.gemma_frames, caps, len(media.gemma_frames))
    return _fill(caps, style_list), f"{(kimi.name if kimi else '-')}+{gemma.name}"


# ---------------------------------------------------------------- describex replica (v23)
# Top-3 recipe: ONE strong model (Qwen) does BOTH stages. Structured 7-category factual scene
# description (vision) -> all 4 styles from that description (text). Same model = no handoff loss.
def _scene_prompt(n_frames: int) -> str:
    return (
        f"You are a precise visual analyst. You are shown {n_frames} representative frames sampled "
        "from a short video. Produce a structured, FACTUAL understanding of the video, covering:\n"
        "1. Scene / Setting — location, venue, environment.\n"
        "2. Subjects — people/animals/objects; appearance, positioning, distinguishing features "
        "(identify by visible features; if ambiguous, say so).\n"
        "3. Actions — activities, movements, interactions across the frames, in order.\n"
        "4. Environment — indoor/outdoor, time of day, weather/season.\n"
        "5. Mood / Tone — atmosphere, lighting, expressions, pacing.\n"
        "6. Key Visual Elements — prominent colors, notable objects, on-screen text (verbatim), transitions.\n"
        "7. Temporal Flow — how the scene progresses from first to last frame; changes / narrative arc.\n\n"
        "Be factual and neutral. Report ONLY what you observe. Do NOT write captions, humor, or "
        "opinion. Do NOT invent names, brands, exact places, dates, or counts you cannot see. "
        "Write clear prose using the numbered categories."
    )


def _styles_from_scene_prompt(scene: str, style_list: list[str]) -> str:
    blocks = "\n\n".join(styles.style_block(s) for s in style_list)
    keys = ", ".join(f'"{s}"' for s in style_list)
    return (
        "You are an expert caption writer. Below is a factual scene description generated from a video.\n\n"
        f"--- SCENE DESCRIPTION ---\n{scene}\n--- END SCENE DESCRIPTION ---\n\n"
        "Write ONE caption for EACH requested style, all describing the SAME scene above.\n"
        "RULES:\n"
        "- Each caption MUST be 2 to 4 sentences long (roughly 40-90 words).\n"
        "- Keep the SAME core facts in every style (subjects, appearance, actions, setting, arc); "
        "only tone/phrasing/jokes change. Funny styles must NOT be shorter in substance than formal.\n"
        "- Humor/sarcasm from the REAL facts only — never invent numbers, names, motives, or dialogue.\n"
        "- Make the four tones genuinely distinct; do not reuse the same joke structure.\n"
        "- Use ONLY facts from the scene description; if a detail is uncertain there, omit it.\n\n"
        f"{blocks}\n\n"
        f"Return ONLY a JSON object with exactly these keys: {keys}. No markdown, no extra text."
    )


async def _scene_description(backend, media) -> str:
    frames = media.send_frames
    return (await backend.chat_vision(frames, _scene_prompt(len(frames)),
                                      temperature=0.2, max_tokens=1000, n=1) or "").strip()


async def caption_describex(task: dict, media, reg, ordered):
    style_list = task["styles"]
    for b in ordered:
        try:
            scene = await _scene_description(b, media)
            if len(scene) < 40:
                raise ModelError("empty/short scene description")
            raw = await b.chat_text(_styles_from_scene_prompt(scene, style_list),
                                    temperature=0.6, max_tokens=1800, n=1)
            caps = schema.coerce_captions(schema.extract_json(raw or ""), style_list)
            if not any(caps.get(s) for s in style_list):
                raise ModelError("empty styles")
            if Config.DEBUG_CONTEXT:
                task["_context"] = {"evidence": scene}
            if "humorous_non_tech" in style_list:
                caps = await _guard_nontech(b, media.send_frames, caps, len(media.send_frames))
            return _fill(caps, style_list), b.name
        except ModelError as e:
            log.warning("[%s] describex %s failed: %s", task["task_id"], b.name, e)
            continue
    log.warning("[%s] describex failed everywhere -> oneshot", task["task_id"])
    return await caption_oneshot(task, media, ordered)


# ---------------------------------------------------------------- native video (v26)
# The model watches the ACTUAL clip (no frame sampling) -> factual description -> 4 styles.
# Native video = maximum true, checkable claims (motion + all actions) = highest accuracy.
_NATIVE_DESC_PROMPT = (
    "You are a precise visual analyst watching a short video clip (roughly 30 seconds to 2 minutes). "
    "Produce a THOROUGH, richly detailed factual description — the more specific TRUE detail you capture "
    "about what is clearly visible, the better. Structure it as:\n"
    "1. Scene / Setting — the location or environment; time of day ONLY if clearly visible; the camera "
    "angle and whether it is static or moving; the lighting (direction, warmth, shadows).\n"
    "2. Main subject — describe the CENTRAL subject EXHAUSTIVELY. Enumerate every clearly visible "
    "attribute as a specific, concrete detail: exact colours, materials and textures, each clothing item "
    "and how it is worn, hairstyle, visible accessories, facial expression, posture, and precisely what "
    "the hands and body are doing. Be as rich and specific as the footage allows — this is the most "
    "important part, so spend the most words here.\n"
    "3. Action & Temporal flow — narrate the clip as a sequence of beats from the first moment to the "
    "last: what happens at the start, what changes through the middle, how it ends. Include the small "
    "movements too (a glance, a pause, a shift in weight, a change in pace or traffic density), in order.\n"
    "4. Notable true details — objects the subject clearly holds or interacts with, and clear foreground "
    "elements, with their colours and positions relative to the subject.\n"
    "5. Background — describe it GENERALLY; do not enumerate or count things you cannot clearly see.\n\n"
    "ANTI-HALLUCINATION RULES — a wrong detail hurts more than a missing one, so follow these strictly:\n"
    "- Signs / on-screen text: quote the words ONLY if large and UNMISTAKABLY legible; otherwise write "
    "'a sign' or 'text' and do NOT guess the words.\n"
    "- Never name a city, country, company, brand, or product unless a clear legible label states it.\n"
    "- People: describe only build, clothing, and hairstyle. NEVER state or guess ethnicity, nationality, "
    "religion, or skin colour.\n"
    "- Do NOT give exact counts of background objects or people; use 'several' / 'a few' / 'a group' if unsure.\n"
    "- Mention a secondary subject ONLY if it is clearly and unambiguously visible for more than an instant.\n"
    "- Treat any watermark, logo, caption overlay, or transition effect as a graphic, never as a real object.\n"
    "- If a detail is uncertain, or only briefly or partly visible, OMIT it rather than guess.\n\n"
    "Go DEEP and specific on the main subject and its action; stay conservative and general about the "
    "background. Report ONLY what is actually visible; write no captions, humour, or opinion. This "
    "description is the SOLE factual source for the captions written afterward — capture every concrete, "
    "checkable detail a writer would need to caption the clip accurately without seeing it."
)


def _native_verify_prompt(description: str) -> str:
    """v29b: SURGICAL redaction — strip only the provable-fabrication categories, keep all real detail.

    The full-prune v29 hurt because it also dropped true descriptive richness (which both the
    judges and, by claim count, real accuracy reward). This version edits ONLY the specific
    high-risk categories (sign text, brands, place names, identity labels, overlays) and keeps
    every other visible detail verbatim, so the fabrication surface shrinks without thinning.
    """
    return (
        "Here is a factual description of a video:\n\n"
        f"{description}\n\n"
        "Return the SAME description, keeping every visible detail — the subject, colours, clothing, "
        "objects, actions, setting, and counts — essentially word for word, changing ONLY the "
        "following where they appear:\n"
        "- Replace any quoted on-screen text or sign wording with just 'a sign' or 'text' (never guess the words).\n"
        "- Replace any brand, company, or product name with a generic term.\n"
        "- Replace any city, country, or landmark name with a generic term (e.g. 'a city street').\n"
        "- Remove any reference to a person's ethnicity, nationality, religion, identity, or skin "
        "colour / complexion (e.g. 'a Black woman' or 'a woman with dark skin' becomes 'a woman'); "
        "keep neutral visible features like hairstyle, clothing, and colours of objects.\n"
        "- Describe any watermark, logo, caption overlay, or transition/dissolve effect generically, "
        "never as a real-world object.\n"
        "Do NOT shorten, summarise, rephrase, or drop any other detail — keep it just as rich, and keep "
        "the same structure and category headings if present. Return ONLY the edited description, no commentary."
    )


def _native_styles_prompt(description: str, style_list: list[str]) -> str:
    blocks = "\n\n".join(styles.style_block(s) for s in style_list)
    keys = ", ".join(f'"{s}"' for s in style_list)
    return (
        "Here is a factual description of a video:\n\n"
        f"{description}\n\n"
        "Write ONE caption for EACH style below, all describing the SAME video.\n"
        "RULES:\n"
        "- About 40-90 words each. Ground every caption in the description above; invent nothing. "
        "The humour/tone comes from HOW you say it, never from adding facts.\n"
        "- Anchor on the main subject and its central action; every claim must be true per the "
        "description. Make the four tones genuinely distinct.\n\n"
        f"{blocks}\n\n"
        f"Return ONLY a JSON object with exactly these keys: {keys}. No markdown, no extra text."
    )


async def caption_native(task: dict, reg, ordered):
    style_list = task["styles"]
    video_url = task.get("video_url", "")
    backend = reg.get(Config.PERCEPTION_BACKEND) or (ordered[0] if ordered else None)
    if not backend or not video_url:
        return _fill({}, style_list), "template"
    try:
        desc = (await backend.chat_video(video_url, _NATIVE_DESC_PROMPT,
                                         temperature=0.2, max_tokens=1500) or "").strip()
        if len(desc) < 40:
            raise ModelError("empty/short native description")
        raw_desc = desc
        if Config.VERIFY_PASS:   # v29b: surgical text redaction — strip sign/brand/place/identity/overlay, keep all real detail
            corrected = (await backend.chat_text(_native_verify_prompt(desc),
                                                 temperature=0.0, max_tokens=1600, n=1) or "").strip()
            if len(corrected) >= 40:
                desc = corrected
        raw = await backend.chat_text(_native_styles_prompt(desc, style_list),
                                      temperature=0.6, max_tokens=1600, n=1)
        caps = schema.coerce_captions(schema.extract_json(raw or ""), style_list)
        if not any(caps.get(s) for s in style_list):
            raise ModelError("empty styles")
        if Config.DEBUG_CONTEXT:
            task["_context"] = {"evidence": desc, "raw_evidence": raw_desc,
                                "changed": bool(Config.VERIFY_PASS and desc != raw_desc)}
        nt = caps.get("humorous_non_tech", "")
        if nt and any(w in nt.lower().split() for w in styles.NON_TECH_BANNED):
            fix = await backend.chat_text(
                _native_styles_prompt(desc, ["humorous_non_tech"]) +
                "\n\nUse ZERO technology or science words.", temperature=0.7, max_tokens=300, n=1)
            fixed = schema.coerce_captions(
                schema.extract_json(fix or ""), ["humorous_non_tech"]).get("humorous_non_tech", "")
            if fixed and not any(w in fixed.lower().split() for w in styles.NON_TECH_BANNED):
                caps["humorous_non_tech"] = fixed
        return _fill(caps, style_list), f"native:{backend.name}"
    except ModelError as e:
        log.warning("[%s] native video failed: %s", task["task_id"], e)
        return _fill({}, style_list), "template"


# ---------------------------------------------------------------- qwen-direct (0.92-architecture replica)
# Top-team recipe (silver-octo HANDOFF board ladder): NO describe, NO selector — ONE multimodal call per
# style with a SHORT imperative persona on a strong VLM (qwen3p7-plus), 4 frames @1024, reasoning off,
# <caption_output> tag extract. Their long-roleplay rewrite tanked 0.92->0.74, so personas here are short +
# imperative and 100% OUR OWN wording (never Quiptionary/HAL-9000 prose). Guardrails: no invented
# sign/brand/place/identity claims + the non-tech lexical ban.
_QDIRECT_SYSTEM = (
    "You are a precise captioning tool. Look at the frames and write exactly ONE caption in the voice "
    "described below. Output ONLY the caption wrapped in <caption_output>...</caption_output> — no "
    "reasoning, no labels, no other text. English only. Describe only what is clearly visible; never "
    "invent sign wording, brand or product names, city / country / landmark names, or a person's "
    "ethnicity or identity."
)
_QDIRECT_PERSONAS = {
    "formal": (
        "Voice: a precise, neutral news-wire caption. State exactly what is shown — the main subject with "
        "concrete attributes (colours, clothing, counts), its central action, and the setting. Present "
        "tense, third person; no opinion, no humour, no first person."
    ),
    "sarcastic": (
        "Voice: dry, deadpan, faintly withering. Aim one unbothered line of irony at a single thing that "
        "is actually on screen. Every fact stays true; only the attitude bites. No exclamation marks."
    ),
    "humorous_tech": (
        "Voice: a developer's wry humour. Land ONE genuinely funny line that maps a software or programming "
        "idea (deploys, bugs, merge conflicts, latency, prod incidents) onto what is literally happening on "
        "screen. At most two tech terms; the joke must land; invent no details."
    ),
    "humorous_non_tech": (
        "Voice: warm everyday-observational comedy. Land ONE funny line about the scene using only ordinary "
        "words — anthropomorphise, or exaggerate a tiny stake. Use ZERO technology, science, or internet words."
    ),
}


def _even_pick(items: list, n: int) -> list:
    if not items or len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _qdirect_prompt(style: str, n_frames: int) -> str:
    persona = _QDIRECT_PERSONAS.get(style, f"Voice: {style.replace('_', ' ')}.")
    return (
        f"You are shown {n_frames} frames sampled in order from a short video clip; read them as one "
        f"moving scene.\n\n{_QDIRECT_SYSTEM}\n\n{persona}\n\nWrite the caption now."
    )


def _extract_caption_output(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"<caption_output>(.*?)</caption_output>", s, re.DOTALL | re.IGNORECASE)
    if m:
        return schema.clean_caption(m.group(1).strip())
    s = re.sub(r"(?is)</?caption_output>", "", s).strip()   # tag opened but not closed, or no tags
    return schema.clean_caption(s)


async def caption_qwen_direct(task: dict, media, reg, ordered):
    style_list = task["styles"]
    frames = _even_pick(getattr(media, "send_frames", None) or [], 4)   # 0.92 recipe: 4 frames @1024
    backend = reg.get(Config.PERCEPTION_BACKEND) or (ordered[0] if ordered else None)
    if not backend or not frames:
        return _fill({}, style_list), "template"

    async def one(style: str):
        try:
            raw = await backend.chat_vision(frames, _qdirect_prompt(style, len(frames)),
                                            temperature=0.7, max_tokens=400, n=1)
            return style, _extract_caption_output(raw)
        except ModelError as e:
            log.warning("[%s] qdirect %s failed: %s", task["task_id"], style, e)
            return style, ""

    results = await asyncio.gather(*[one(s) for s in style_list])
    caps = {s: c for s, c in results if c}
    nt = caps.get("humorous_non_tech", "")   # one regen if the non-tech style leaked a banned word
    if nt and any(w in nt.lower().split() for w in styles.NON_TECH_BANNED):
        try:
            raw = await backend.chat_vision(
                frames, _qdirect_prompt("humorous_non_tech", len(frames)) +
                "\n\nUse ZERO technology or science words — only plain everyday language.",
                temperature=0.9, max_tokens=400, n=1)
            c = _extract_caption_output(raw)
            if c and not any(w in c.lower().split() for w in styles.NON_TECH_BANNED):
                caps["humorous_non_tech"] = c
        except ModelError:
            pass
    if not any(caps.get(s) for s in style_list):
        return _fill({}, style_list), "template"
    return _fill(caps, style_list), f"qdirect:{backend.name}"


# ---------------------------------------------------------------- context->gemma (enriched, Gemma-prize)
# Two-stage, Gemma writes the FINAL captions (keeps the $3k Gemma-prize). A strong perceiver watches the
# whole clip -> structured scene + TEMPORAL description; Whisper adds audio ONLY if real speech is present
# (music/ambient is skipped — audio poisoned music-only clips before). Gemma then gets ALL of it PLUS the
# actual frames + our gold personas/guardrails, so there is no lossy text-only handoff. PERCEPTION_BACKEND
# is minimax native today; swap to gemini (native video describe) once a GEMINI_API_KEY is set.
# Shared creator persona across ALL styles: obsessed with the ONE perfect caption (attention + TRUTH),
# hard anti-clickbait. The per-style block below supplies the voice; this supplies the drive + honesty.
_CREATOR_COMMON = (
    "You are a video creator obsessed with the ONE perfect caption for this clip — the line that makes "
    "someone stop and watch. But you are brutally honest: a caption that exaggerates or invents anything is "
    "worthless, because viewers feel the lie. Every caption you write must be:\n"
    "- TRUE to the video: only what is clearly shown. Never invent a detail, number, place, brand, a "
    "person's identity or skin colour, or a sign's words.\n"
    "- A REAL HOOK: the pull comes from one sharp, specific thing actually in this video — never from hype.\n"
    "- NEVER clickbait: no '1 vs 10,000', no 'I spent 100 days', no 'you won't believe', no fake stakes, "
    "no ALL-CAPS, no emoji, no listicle formulas.\n"
    "- In the EXACT requested voice below — the style always wins the register."
)


def _ctxgemma_prompt(description: str, transcript: str, style_list: list[str]) -> str:
    blocks = "\n\n".join(styles.style_block(s) for s in style_list)
    keys = ", ".join(f'"{s}"' for s in style_list)
    audio_block = ""
    if transcript:
        audio_block = ("\nAUDIO (speech heard in the clip — use only if it adds a TRUE detail; ignore if "
                       f"it is music or noise):\n\"{transcript}\"\n")
    ctx = description if description else "(no separate analysis available — rely on the frames)"
    return (
        "You are shown the actual FRAMES of a short video clip, plus a factual analysis of the same clip. "
        "Treat them as a single moving scene.\n\n"
        f"--- SCENE ANALYSIS ---\n{ctx}\n--- END ---\n"
        f"{audio_block}\n"
        "Write ONE caption for EACH requested style below, all describing the SAME video.\n\n"
        "HARD RULES:\n"
        "- Be DETAILED but TIGHT: about 90-120 words per caption (4-6 sentences). Cover the main "
        "subject(s) and their concrete attributes, the actions in order, and the setting. Rich TRUE "
        "detail raises accuracy — but do NOT ramble past ~120 words.\n"
        "- Ground EVERY detail ONLY in the frames + analysis; never invent names, brands, exact locations, "
        "or counts you cannot actually see — a wrong detail hurts more than a missing one.\n"
        "- Write the WHOLE caption in the requested style's voice, start to finish, never just one clause. "
        "English only. Make the four voices genuinely distinct.\n\n"
        f"{blocks}\n\n"
        f"Return ONLY a JSON object with exactly these keys: {keys}. No markdown, no extra text."
    )


async def caption_ctxgemma(task: dict, media, reg, ordered):
    style_list = task["styles"]
    video_url = task.get("video_url", "")
    frames = getattr(media, "send_frames", None) or []
    perceiver = reg.get(Config.PERCEPTION_BACKEND)          # minimax native (or gemini once keyed)
    writer = reg.get(Config.STYLIZE_BACKEND) or reg.get("hf")  # Gemma writes the final captions
    if not writer or (not frames and not video_url):
        return _fill({}, style_list), "template"
    # 1. scene + temporal description (perceiver watches the whole clip)
    desc = ""
    raw_desc = ""
    if perceiver:
        try:
            if Config.PERCEPTION_BACKEND == "gemini":   # Gemini (OpenAI-compat) reads FRAMES, not video_url
                desc = (await perceiver.chat_vision(frames, _NATIVE_DESC_PROMPT,
                                                    temperature=0.2, max_tokens=1500) or "").strip()
            elif video_url:                             # minimax etc. watch the native video
                desc = (await perceiver.chat_video(video_url, _NATIVE_DESC_PROMPT,
                                                   temperature=0.2, max_tokens=1500) or "").strip()
            raw_desc = desc
            if desc and Config.VERIFY_PASS:   # backup guardrail: strip any sign/brand/place/identity that slipped
                corrected = (await perceiver.chat_text(_native_verify_prompt(desc),
                                                       temperature=0.0, max_tokens=1600) or "").strip()
                if len(corrected) >= 40:
                    desc = corrected
        except ModelError as e:
            log.warning("[%s] ctxgemma describe failed: %s", task["task_id"], e)
    # 2. audio, speech-guarded (skip music/noise)
    transcript = ""
    if Config.USE_AUDIO and getattr(media, "audio_path", None):
        try:
            t = (await transcribe_audio(writer.client, media.audio_path) or "").strip()
            if len(t.split()) >= 3:
                transcript = t
        except Exception:
            pass
    # 3. Gemma writes from frames + context + gold prompts, per-style temperature (formal cold, witty hot)
    for b in [writer] + [x for x in ordered if x is not writer]:
        try:
            caps = await _gen_by_temp(b, frames, style_list,
                                      lambda grp: _ctxgemma_prompt(desc, transcript, grp), max_tokens=1200)
            if not any(caps.get(s) for s in style_list):
                raise ModelError("empty styles")
            if "humorous_non_tech" in style_list:
                caps = await _guard_nontech(b, frames, caps, len(frames))
            if Config.DEBUG_CONTEXT:
                task["_context"] = {"evidence": desc, "raw_evidence": raw_desc, "audio": transcript}
            return _fill(caps, style_list), f"ctxgemma:{b.name}"
        except ModelError as e:
            log.warning("[%s] ctxgemma writer %s failed: %s", task["task_id"], b.name, e)
            continue
    return _fill({}, style_list), "template"


# ---------------------------------------------------------------- dispatch
async def caption_clip(task: dict, media, reg, mode: str):
    ordered = ordered_backends(reg)
    if mode == "bestofn":
        return await caption_bestofn(task, media, reg, ordered)
    if mode == "ctxgemma":
        return await caption_ctxgemma(task, media, reg, ordered)
    if mode == "qwen_direct":
        return await caption_qwen_direct(task, media, reg, ordered)
    if mode == "native_video":
        return await caption_native(task, reg, ordered)
    if mode == "describex":
        return await caption_describex(task, media, reg, ordered)
    if mode == "kimigemma":
        return await caption_kimigemma(task, media, reg, ordered)
    if mode == "twopass":
        return await caption_twopass(task, media, reg, ordered)
    if mode == "twostage":
        return await caption_twostage(task, media, reg, ordered)
    return await caption_oneshot(task, media, ordered)
