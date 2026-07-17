"""Output schema + robust JSON extraction (BLUEPRINT R-09, R-10, R-11, F-02, F-09).

The results file is a JSON array of {task_id, captions{style: str}}. This module
is the single gatekeeper that guarantees we never emit malformed output or drop a
requested style.
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON value from a model response.

    Models wrap JSON in ```json fences, add prose, or emit trailing commas. Try
    increasingly permissive strategies; return None if nothing parses.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    s = text.strip()
    # 1) straight parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) strip code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            s = fence.group(1).strip()
    # 3) grab the first balanced {...} or [...] blob
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    blob = s[start : i + 1]
                    try:
                        return json.loads(blob)
                    except Exception:
                        # last resort: kill trailing commas
                        try:
                            return json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
                        except Exception:
                            break
    return None


def clean_caption(text: Any) -> str:
    """Normalize one caption string: strip fences, meta-prefixes, quotes, whitespace."""
    if text is None:
        return ""
    s = str(text).strip()
    # drop a leading "Formal: " / "Here's a sarcastic caption:" style prefix
    s = re.sub(r"^\s*(here'?s?|caption|formal|sarcastic|humorous[_ ]?\w*)\b[^A-Za-z0-9]{0,4}:?\s*",
               "", s, flags=re.IGNORECASE)
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def coerce_captions(raw: Any, styles: list[str]) -> dict[str, str]:
    """Turn an arbitrary model payload into {style: caption} for the requested styles.

    Accepts {"captions": {...}} or a flat {style: cap} dict. Missing/blank styles
    are returned absent so the caller can decide how to fill them (F-02).
    """
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        src = raw.get("captions") if isinstance(raw.get("captions"), dict) else raw
        if isinstance(src, dict):
            # case-insensitive style key match
            lower = {str(k).lower(): v for k, v in src.items()}
            for st in styles:
                val = lower.get(st.lower())
                cap = clean_caption(val)
                if cap:
                    out[st] = cap
    return out


def validate_results(results: list[dict], tasks: list[dict]) -> list[str]:
    """Return a list of schema problems (empty = valid). BLUEPRINT F-09 / T-09."""
    problems: list[str] = []
    if not isinstance(results, list):
        return ["results is not a list"]
    by_id = {}
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            problems.append(f"entry[{i}] not an object")
            continue
        tid = r.get("task_id")
        if not isinstance(tid, str) or not tid:
            problems.append(f"entry[{i}] missing/invalid task_id")
            continue
        by_id[tid] = r
        caps = r.get("captions")
        if not isinstance(caps, dict):
            problems.append(f"task {tid}: captions missing/not an object")
            continue
    for t in tasks:
        tid = t["task_id"]
        r = by_id.get(tid)
        if r is None:
            problems.append(f"task {tid}: absent from results")
            continue
        caps = r.get("captions", {})
        for st in t.get("styles", []):
            v = caps.get(st)
            if not isinstance(v, str) or not v.strip():
                problems.append(f"task {tid}: style '{st}' missing/empty")
    return problems
