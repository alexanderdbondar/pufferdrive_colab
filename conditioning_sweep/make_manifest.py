#!/usr/bin/env python3
"""Build manifest.csv for a converted PufferDrive split.

The evaluator reads manifest.csv from the split directory to find the ego agent
in each scene and to rank scenes. interactivity_score is the number of other
agents that come within 10 m of the ego.

Usage:
    python make_manifest.py --json-dir /content/data/processed/validation_subset \
                            --split-dir /content/maps/validation
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def trajectory(obj):
    pos = obj.get("position") or []
    if not pos:
        return None, None
    xy = np.array([[p["x"], p["y"]] for p in pos], dtype=float)
    valid = np.array(obj.get("valid") or [True] * len(pos), dtype=bool)
    return xy, valid


def score_scene(scene, ego_idx):
    objects = scene["objects"]
    if ego_idx >= len(objects):
        return 0, -1.0
    ego_xy, ego_valid = trajectory(objects[ego_idx])
    if ego_xy is None:
        return 0, -1.0

    near, min_dist = 0, float("inf")
    for j, obj in enumerate(objects):
        if j == ego_idx:
            continue
        other_xy, other_valid = trajectory(obj)
        if other_xy is None:
            continue
        length = min(len(ego_xy), len(other_xy))
        both = ego_valid[:length] & other_valid[:length]
        if not both.any():
            continue
        dist = np.linalg.norm(ego_xy[:length][both] - other_xy[:length][both], axis=1)
        min_dist = min(min_dist, float(dist.min()))
        if dist.min() < 10.0:
            near += 1
    return near, (round(min_dist, 2) if min_dist < float("inf") else -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    args = ap.parse_args()

    # process_all_maps() numbers binaries in sorted-path order.
    files = sorted(Path(args.json_dir).rglob("*.json"))
    rows = []
    for i, path in enumerate(files):
        scene = json.load(open(path))
        ego_idx = int(scene.get("metadata", {}).get("sdc_track_index", 0))
        near, min_dist = score_scene(scene, ego_idx)
        rows.append({
            "new_filename": f"map_{i:06d}.bin",
            "original_filename": path.name,
            "ego_agent_idx": ego_idx,
            "interactivity_score": near,
            "map_id": i,
            "min_distance_m": min_dist,
        })

    out = Path(args.split_dir) / "manifest.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda r: (-r["interactivity_score"], r["min_distance_m"]))
    print(f"Wrote {out} ({len(rows)} scenes)")
    print("Most interactive scenes:")
    for r in rows[:10]:
        print(f"  map {r['map_id']:4d}  neighbours<10m={r['interactivity_score']:2d}  "
              f"closest={r['min_distance_m']:5.1f} m")


if __name__ == "__main__":
    main()
