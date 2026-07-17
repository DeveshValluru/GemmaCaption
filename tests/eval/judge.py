"""LLM-judge replica: scores a results.json against Kimi references.

Usage:  python tests/eval/judge.py <results.json> [label]
Scores each caption: accuracy (vs reference description) and style (vs official
definition), 0-1 each, mirroring the competition rubric. Judge = gpt-oss-120b via
HF router (different family from both Gemma and Kimi -> less self-preference).
Prints per-style and overall means; appends a row to tests/eval/scoreboard.csv.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

STYLE_DEFS = {
    "formal": "Professional, objective, factual tone.",
    "sarcastic": "Dry, ironic, lightly mocking.",
    "humorous_tech": "Funny, with technology or programming references.",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon.",
}
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b:fastest")
HERE = os.path.dirname(os.path.abspath(__file__))


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = os.path.join(HERE, "..", "..", ".env")
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith("HF_TOKEN"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no HF_TOKEN")


def judge_clip(tok, ref, captions):
    listing = "\n".join(
        f'{i+1}. style="{s}" (definition: {STYLE_DEFS[s]})\n   caption: "{captions[s]}"'
        for i, s in enumerate(k for k in STYLE_DEFS if k in captions))
    prompt = (
        "You are an strict evaluation judge for video captions.\n\n"
        f"REFERENCE DESCRIPTION of the video (ground truth):\n{ref}\n\n"
        f"CAPTIONS TO GRADE:\n{listing}\n\n"
        "For EACH caption give two scores from 0.0 to 1.0:\n"
        "- accuracy: are the caption's claims CORRECT per the reference, and does it capture the "
        "gist (main subject and main action)? Start from 1.0. Subtract heavily for any claim that "
        "is wrong or contradicted by the reference; subtract moderately if the main subject or "
        "main action is missing/wrong. Do NOT penalize brevity or omitted minor details — a short "
        "caption whose claims are all correct and covers the gist scores 0.9-1.0.\n"
        "- style: how well the caption matches its requested style definition. Penalize mixed "
        "or generic register.\n\n"
        'Return ONLY JSON: {"<style>": {"accuracy": 0.0, "style": 0.0}, ...}'
    )
    r = httpx.post("https://router.huggingface.co/v1/chat/completions",
                   json={"model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": 2500,
                         "messages": [{"role": "user", "content": prompt}]},
                   headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # reasoning models may put the answer in content, or burn tokens in reasoning
    txt = (msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")
    m = re.findall(r"\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\}", txt, re.DOTALL)
    for blob in sorted(m, key=len, reverse=True):
        try:
            d = json.loads(blob)
            if isinstance(d, dict) and any(k in STYLE_DEFS for k in d):
                return d
        except Exception:
            continue
    return {}


def main():
    results_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(results_path)
    tok = hf_token()
    refs = {json.loads(l)["task_id"]: json.loads(l)["reference"]
            for l in open(os.path.join(HERE, "refs.jsonl"), encoding="utf-8")}
    results = {r["task_id"]: r["captions"] for r in json.load(open(results_path, encoding="utf-8"))}

    rows = []
    def work(tid):
        for attempt in range(3):
            try:
                return tid, judge_clip(tok, refs[tid], results[tid])
            except Exception as e:
                if attempt == 2:
                    print(f"[judge fail] {tid}: {e}", file=sys.stderr)
                    return tid, {}
                time.sleep(3 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=4) as ex:
        for tid, scores in ex.map(work, [t for t in refs if t in results]):
            for st, sc in (scores or {}).items():
                if st in STYLE_DEFS and isinstance(sc, dict):
                    try:
                        rows.append((tid, st, float(sc["accuracy"]), float(sc["style"])))
                    except (KeyError, TypeError, ValueError):
                        pass

    if not rows:
        print("NO SCORES")
        return 1
    accs = [r[2] for r in rows]; stys = [r[3] for r in rows]
    overall = (sum(accs) + sum(stys)) / (2 * len(rows))
    print(f"\n=== {label} ===  ({len(rows)} captions judged)")
    print(f"overall={overall:.3f}  accuracy={sum(accs)/len(accs):.3f}  style={sum(stys)/len(stys):.3f}")
    for st in STYLE_DEFS:
        sr = [r for r in rows if r[1] == st]
        if sr:
            print(f"  {st:20s} acc={sum(r[2] for r in sr)/len(sr):.3f} sty={sum(r[3] for r in sr)/len(sr):.3f}")
    worst = sorted(rows, key=lambda r: r[2] + r[3])[:4]
    for w in worst:
        print(f"  WORST: {w[0]}/{w[1]} acc={w[2]:.2f} sty={w[3]:.2f}")
    with open(os.path.join(HERE, "scoreboard.csv"), "a", encoding="utf-8") as f:
        f.write(f"{label},{overall:.4f},{sum(accs)/len(accs):.4f},{sum(stys)/len(stys):.4f},{len(rows)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
