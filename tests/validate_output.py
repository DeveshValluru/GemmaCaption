"""Standalone output validator for acceptance tests (BLUEPRINT T-01/02/07/08/09).

Usage: python validate_output.py <tasks.json> <results.json>
Exit 0 = valid. Checks schema (R-11), completeness (R-10), non-empty English (R-12),
and flags obvious style problems (tech words in humorous_non_tech).
"""
import json
import sys

BANNED_NON_TECH = {
    "app", "ai", "algorithm", "wifi", "update", "download", "battery", "screen", "code",
    "robot", "startup", "notification", "internet", "digital", "online", "software",
    "server", "api", "computer", "laptop", "website", "cloud", "smartphone",
}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    tasks = load(sys.argv[1])
    results = load(sys.argv[2])
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [tasks])

    by_id = {}
    errors, warnings = [], []
    if not isinstance(results, list):
        print("FAIL: results is not a JSON array")
        return 1
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("task_id"), str):
            by_id[r["task_id"]] = r.get("captions", {})

    n_caps = 0
    for t in tasks:
        tid = t.get("task_id")
        caps = by_id.get(tid)
        if caps is None:
            errors.append(f"{tid}: missing from results")
            continue
        for st in t.get("styles", []):
            v = caps.get(st)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{tid}/{st}: missing or empty")
                continue
            n_caps += 1
            # English-ish: low non-ASCII ratio
            non_ascii = sum(1 for c in v if ord(c) > 127)
            if non_ascii / max(1, len(v)) > 0.15:
                warnings.append(f"{tid}/{st}: high non-ASCII ratio (English? R-12)")
            words = len(v.split())
            if words > 60:
                warnings.append(f"{tid}/{st}: very long ({words} words)")
            if st == "humorous_non_tech":
                hits = {w for w in BANNED_NON_TECH if w in v.lower().split()}
                if hits:
                    warnings.append(f"{tid}/{st}: tech words in non-tech style: {sorted(hits)}")

    print(f"tasks={len(tasks)} captions={n_caps} errors={len(errors)} warnings={len(warnings)}")
    for e in errors:
        print("  ERROR:", e)
    for w in warnings[:20]:
        print("  warn :", w)
    if errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS" + (" (with warnings)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
