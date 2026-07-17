"""Generate a synthetic 12-task input (T-02) by rotating the 3 example clips."""
import json
import os

CLIPS = [
    "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
]
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

tasks = [{"task_id": f"s{i:02d}", "video_url": CLIPS[i % len(CLIPS)], "styles": STYLES}
         for i in range(12)]

out_dir = os.path.join(os.path.dirname(__file__), "scale_input")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "tasks.json"), "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)
print(f"wrote {len(tasks)} tasks to {out_dir}/tasks.json")
