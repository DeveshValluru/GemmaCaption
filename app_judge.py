"""StyleCap — caption judge.

Review one version at a time: play each clip, read its full (uncompressed) captions,
and hit **Regenerate** to re-run that version's Docker image live on a single clip.

Run:  streamlit run app_judge.py
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

BASE = Path(__file__).parent
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
STYLE_LABEL = {
    "formal": "📋 Formal",
    "sarcastic": "🙄 Sarcastic",
    "humorous_tech": "💻 Humorous · tech",
    "humorous_non_tech": "😄 Humorous · everyday",
}
NON_TECH_BANNED = {
    "app", "ai", "algorithm", "wifi", "update", "download", "battery", "screen", "code",
    "robot", "startup", "notification", "internet", "digital", "online", "software",
    "server", "pixel", "smartphone", "laptop", "computer", "website", "cloud", "api",
}
VERSION_META = {
    "v2": "Gemma two-stage, 5 frames — real: 0.70",
    "v4": "Gemma two-stage, grids + best-of-N — real: 0.73",
    "v5": "Gemma two-stage + Whisper audio — real: 0.67",
    "v10": "Kimi one-shot, SHORT captions — floor ~0.77",
    "v11": "Kimi→Gemma two-stage + audio — real: 0.63",
    "v13": "Kimi→Gemma two-stage, 24 frames — real: 0.60",
    "v14": "Gemma→Gemma two-stage, 5 frames — submitted",
    "v15": "Gemma one-shot, SHORT captions",
    "v16": "Gemma ONE-SHOT · LONG detailed captions — $3k Gemma-prize play",
    "v17": "Kimi ONE-SHOT · LONG detailed captions — leaderboard / accuracy play",
}

st.set_page_config(page_title="StyleCap Judge", layout="wide")


def version_key(label: str) -> str:
    m = re.match(r"(v\d+)", label)
    return m.group(1) if m else label


def _read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_tasks(path: str) -> dict:
    data = _read_json(path)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    return {t["task_id"]: t for t in tasks}


@st.cache_data(show_spinner=False)
def load_results(path: str) -> dict:
    data = _read_json(path)
    items = data.get("results", data) if isinstance(data, dict) else data
    return {r["task_id"]: r for r in items if isinstance(r, dict)}   # full record: captions + context


def show_context(rec: dict):
    """Render the Pass-1 grounding context (evidence, subject-verify, hooks, uncertain)."""
    ctx = rec.get("context")
    if not ctx:
        return
    with st.container(border=True):
        st.markdown("### 🧠 Pass 1 context (grounding)")
        subj = ctx.get("subject") or {}
        conf = str(subj.get("confidence", "?")).lower()
        badge = "🟢 high" if conf == "high" else ("🟠 low" if conf == "low" else "⚪ ?")
        st.markdown(
            f"**Subject:** {subj.get('what', '?')} &nbsp;|&nbsp; confidence: {badge}<br>"
            f"<span style='color:#888'>features: {subj.get('features', '—')}</span>",
            unsafe_allow_html=True,
        )
        if ctx.get("evidence"):
            st.markdown(f"**Evidence:** {ctx['evidence']}")
        if ctx.get("hooks"):
            st.markdown("**🎭 Comedic hooks:** " + " · ".join(map(str, ctx["hooks"])))
        if ctx.get("uncertain"):
            st.markdown("**⚠ Uncertain (do-not-assert):** " + " · ".join(map(str, ctx["uncertain"])))
        if ctx.get("audio"):
            st.markdown(f"**🔊 Audio:** {ctx['audio']}")


def discover_results() -> dict:
    found = {}
    for p in sorted(glob.glob(str(BASE / "tests" / "out_*" / "results.json"))):
        found[Path(p).parent.name.replace("out_", "")] = p
    return found


def banned_hits(cap: str) -> list:
    toks = cap.lower().replace(".", " ").replace(",", " ").split()
    return sorted({w for w in toks if w in NON_TECH_BANNED})


def retry_clip(tag: str, task: dict, timeout: int = 240):
    """Re-run the local Docker image stylecap:<tag> on a single clip; return fresh captions."""
    tmp = Path(tempfile.mkdtemp(prefix="scap_"))
    (tmp / "in").mkdir()
    (tmp / "out").mkdir()
    (tmp / "in" / "tasks.json").write_text(json.dumps([task]), encoding="utf-8")
    indir = str(tmp / "in").replace("\\", "/")
    outdir = str(tmp / "out").replace("\\", "/")
    cmd = ["docker", "run", "--rm", "-e", "DEBUG_CONTEXT=1", "-v", f"{indir}:/input:ro",
           "-v", f"{outdir}:/output", f"stylecap:{tag}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    rp = tmp / "out" / "results.json"
    if rp.exists():
        data = _read_json(str(rp))
        if data:
            return data[0], None   # full record: captions + context + pass1
    tail = (p.stderr or p.stdout or "no output").strip()
    return None, tail[-600:]


def show_caption(style: str, cap: str):
    hits = banned_hits(cap) if style == "humorous_non_tech" else []
    wc = len(cap.split()) if cap else 0
    st.markdown(f"##### {STYLE_LABEL[style]} &nbsp;<span style='color:#999;font-size:.8em'>({wc} words)</span>",
                unsafe_allow_html=True)
    if hits:
        st.markdown(f"<span style='color:#e06c00'>⚠ tech word in non-tech style: {', '.join(hits)}</span>",
                    unsafe_allow_html=True)
    st.write(cap or "—")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎬 StyleCap Judge")

task_files = sorted(glob.glob(str(BASE / "tests" / "*" / "tasks.json")))
default_tasks = str(BASE / "tests" / "scale_input" / "tasks.json")
task_path = st.sidebar.selectbox(
    "Clips (tasks.json)", task_files,
    index=task_files.index(default_tasks) if default_tasks in task_files else 0,
)

results_map = discover_results()
if not results_map:
    st.error("No results found. Run a container against tests/scale_input first.")
    st.stop()

opts = list(results_map.keys())
default_v = next((v for v in ["v21scale", "v20scale", "v16scale"] if v in opts), opts[-1])
version = st.sidebar.selectbox("Version (one at a time)", opts, index=opts.index(default_v))
st.sidebar.info(f"**{version}** — {VERSION_META.get(version_key(version), 'a StyleCap run')}")

view = st.sidebar.radio("View", ["Single clip + video", "All clips (compact)"])

tasks = load_tasks(task_path)
res = load_results(results_map[version])
if "retry" not in st.session_state:
    st.session_state.retry = {}

# ---------------------------------------------------------------- main
st.title("StyleCap — caption judge")
st.caption(f"Reviewing **{version}** · {len(tasks)} clips · {VERSION_META.get(version_key(version), '')}")

if view == "Single clip + video":
    ids = list(tasks.keys())
    tid = st.selectbox("Clip", ids, format_func=lambda t: f"{t} — {tasks[t]['video_url'].split('/')[-1]}")
    task = tasks[tid]
    col_v, col_c = st.columns([2, 3], gap="large")
    with col_v:
        try:
            st.video(task["video_url"])
        except Exception:
            st.markdown(f"[Open clip]({task['video_url']})")
        st.caption(task["video_url"].split("/")[-1])
        tag = version_key(version)
        if st.button(f"🔄 Regenerate this clip with `stylecap:{tag}` (live Docker run)"):
            with st.spinner(f"Running stylecap:{tag} on {tid} … (~10–60s)"):
                rec_new, err = retry_clip(tag, task)
            if rec_new:
                st.session_state.retry[f"{tag}:{tid}"] = rec_new
                st.success("Fresh output below ↓")
            else:
                st.error(f"Retry failed:\n\n```\n{err}\n```")
    with col_c:
        fresh = st.session_state.retry.get(f"{version_key(version)}:{tid}")
        rec = fresh or res.get(tid, {})
        if fresh:
            st.success("🔄 Showing freshly regenerated output")
        show_context(rec)
        caps = rec.get("captions", {})
        for style in STYLES:
            show_caption(style, caps.get(style, ""))
        p1 = rec.get("pass1_captions")
        if p1:
            with st.expander("🔍 Pass 1 captions (before refine) — compare to final above"):
                for style in STYLES:
                    st.markdown(f"**{STYLE_LABEL[style]}**")
                    st.write(p1.get(style, "—"))
        with st.expander("🧾 Raw result JSON"):
            st.json(rec)
else:
    for tid, task in tasks.items():
        with st.expander(f"🎬 {tid} — {task['video_url'].split('/')[-1]}", expanded=False):
            rec = res.get(tid, {})
            show_context(rec)
            caps = rec.get("captions", {})
            for style in STYLES:
                show_caption(style, caps.get(style, ""))
            with st.expander("🧾 Raw result JSON"):
                st.json(rec)
