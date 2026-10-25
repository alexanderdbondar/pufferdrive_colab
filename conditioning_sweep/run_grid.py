#!/usr/bin/env python3
"""Run many (knob, value) evaluations in parallel, in small chunks.

A full 100-scene run takes about 15 minutes. Twenty-five scene chunks limit the
loss if the machine restarts, and finished chunks are skipped on the next start.

Usage:
    python run_grid.py --workers 3 --chunk 25
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def top_maps(manifest, n):
    rows = list(csv.DictReader(open(manifest)))
    rows.sort(key=lambda r: -float(r.get("interactivity_score") or 0))
    return [str(r["map_id"]) for r in rows[:n]]


def existing_maps(pattern):
    rows = []
    for path in glob.glob(pattern, recursive=True):
        rows.extend(csv.DictReader(open(path)))
    return sorted({str(r["map_id"]) for r in rows}, key=int)


def chunks(items, size):
    for i in range(0, len(items), size):
        yield i // size, items[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--repo", default=os.path.join(ROOT, "behaviorbench"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "maps"))
    args = ap.parse_args()

    manifest = os.path.join(args.data_root, args.split, "manifest.csv")
    if not os.path.isfile(manifest):
        sys.exit(f"missing {manifest}")

    # The 34 scenes used for the dial comparison. alpha_center_bias was only run
    # on that set, so it marks which scenes are in it.
    small = existing_maps(os.path.join(args.out, "alpha_center_bias", "*", "**", "per_map.csv"))
    if not small:
        small = top_maps(manifest, 40)
    large = top_maps(manifest, 100)

    knobs_small = {
        "goal_speed": [5, 20, 30],
        "alpha_center_bias": [-0.5, 0.5],
        "alpha_l_center": [0.00025, 0.0075],
        "alpha_comfort": [0.0, 0.1],
        "alpha_boundary": [0, 1, 3],
    }
    knobs_large = {
        "alpha_collision": [0, 1, 3],
        "goal_speed": [5, 30],
    }

    jobs = []
    for knob, values in knobs_small.items():
        for value in values:
            for idx, part in chunks(small, args.chunk):
                jobs.append((knob, value, part, f"chunk_{idx}", small))
    for knob, values in knobs_large.items():
        for value in values:
            for idx, part in chunks(large, args.chunk):
                jobs.append((knob, value, part, f"chunk_{idx}", large))

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)

    def run(job):
        knob, value, part, chunk_name, _ = job
        out_dir = os.path.join(args.out, knob, f"{value:g}", chunk_name)
        done = glob.glob(os.path.join(out_dir, "**", "per_map.csv"), recursive=True)
        if done:
            return f"skip {knob}={value:g} {chunk_name}"
        log_path = os.path.join(ROOT, "logs", f"{knob}_{value:g}_{chunk_name}.log")
        cmd = [
            sys.executable, "-m", "pufferlib.ocean.benchmark.eval",
            "--eval.split", args.split,
            "--map-ids", ",".join(part),
            "--planner.type", "conditioned_normal",
            "--planner.conditioned-normal.weights-path", "weights/conditioned_ppo.pt",
            "--planner.conditioned-normal.device", "cpu",
            "--planner.conditioned-normal.creward." + knob.replace("_", "-"), str(value),
            "--traffic.type", "idm",
            "--output-dir", out_dir,
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            DRIVE_BINARIES_DATA_ROOT=args.data_root,
        )
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, cwd=args.repo, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        return f"{'ok  ' if result.returncode == 0 else 'FAIL'} {knob}={value:g} {chunk_name}"

    print(f"{len(jobs)} jobs, {args.workers} workers, "
          f"{len(small)} small scenes, {len(large)} large scenes")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
