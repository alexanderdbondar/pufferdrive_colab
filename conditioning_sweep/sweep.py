#!/usr/bin/env python3
"""Sweep one reward weight and measure what changes.

The policy is frozen. Each run pins every agent to one value of one weight and
calls BehaviorBench's evaluator, so the numbers come from the same code as the
paper.

Example:
    DRIVE_BINARIES_DATA_ROOT=/content/maps \\
    python sweep.py --knob alpha_collision --values 0 0.75 1.5 2.25 3.0

Output:
    runs/<knob>/<value>/.../per_map.csv   raw per-scene results
    runs/<knob>/results.csv               aggregated table
    runs/<knob>/plot.png                  collision rate and speed change
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

# Keys accepted by the evaluator's --planner.conditioned-normal.creward.* flags.
KNOBS = {
    "alpha_collision": "crash aversion",
    "alpha_boundary": "road-edge aversion",
    "alpha_comfort": "comfort / smoothness",
    "alpha_l_align": "stay aligned with lane",
    "alpha_vel_align": "keep speed while aligned",
    "alpha_l_center": "stay centred in lane",
    "alpha_center_bias": "left/right lane bias",
    "alpha_reverse": "avoid reversing",
    "goal_speed": "target speed",
    "delta_goal": "goal distance",
}

# Columns collected from per_map.csv. Percent columns are already stored as
# percentages (for example "5.56%"), so they are not scaled again.
METRICS = {
    "collision_rate": "collision %",
    "offroad_rate": "offroad %",
    "mean_speed_mps": "speed (m/s)",
    "goal_reached": "goal reached %",
    "num_steps": "episode steps",
    "score_comfort": "comfort score",
    "score_l_align": "lane-align score",
    "score_l_center": "lane-centre score",
}


def parse_percent(value):
    return float(str(value).replace("%", "").strip())


def collect_rows(value_dir):
    """Read every per_map.csv under a value directory. The newest row wins per map_id."""
    hits = glob.glob(os.path.join(value_dir, "**", "per_map.csv"), recursive=True)
    hits.sort(key=os.path.getmtime)
    by_map = {}
    for path in hits:
        for row in csv.DictReader(open(path)):
            by_map[str(row.get("map_id", ""))] = row
    return list(by_map.values())


def metrics_from_rows(rows):
    out = {k: [] for k in METRICS}
    for row in rows:
        for key in METRICS:
            raw = row.get(key, "")
            if raw == "":
                continue
            if key in ("collision_rate", "offroad_rate"):
                out[key].append(parse_percent(raw))
            elif key == "goal_reached":
                out[key].append(100.0 if str(raw).strip() == "True" else 0.0)
            else:
                out[key].append(float(raw))
    return out


def run_one(args, value, out_dir):
    cmd = [
        sys.executable, "-m", "pufferlib.ocean.benchmark.eval",
        "--eval.split", args.split,
        "--map-ids", args.map_ids,
        "--planner.type", "conditioned_normal",
        "--planner.conditioned-normal.weights-path", args.weights,
        "--planner.conditioned-normal.device", args.device,
        "--planner.conditioned-normal.creward." + args.knob.replace("_", "-"), str(value),
        "--traffic.type", args.traffic,
        "--output-dir", out_dir,
    ]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=args.repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="alpha_collision", choices=sorted(KNOBS))
    ap.add_argument("--values", type=float, nargs="+",
                    default=[0.0, 0.75, 1.5, 2.25, 3.0])
    ap.add_argument("--map-ids", default="", help="comma list or start-end; default: --maps most interactive")
    ap.add_argument("--maps", type=int, default=40)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--traffic", default="idm")
    ap.add_argument("--weights", default="weights/conditioned_ppo.pt")
    ap.add_argument("--repo", default="behaviorbench", help="path to the BehaviorBench clone")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip evaluation, just rebuild results.csv and the plot")
    args = ap.parse_args()

    if not os.environ.get("DRIVE_BINARIES_DATA_ROOT"):
        sys.exit("Set DRIVE_BINARIES_DATA_ROOT to the folder containing <split>/")

    args.repo = os.path.abspath(args.repo)
    if not os.path.isdir(args.repo):
        sys.exit(f"BehaviorBench repo not found at {args.repo}")

    # Pick the most interactive scenes from the split's manifest, if present.
    if not args.map_ids:
        manifest = os.path.join(os.environ["DRIVE_BINARIES_DATA_ROOT"], args.split, "manifest.csv")
        if not os.path.isfile(manifest):
            sys.exit("Pass --map-ids, or provide manifest.csv with an interactivity_score column")
        rows = list(csv.DictReader(open(manifest)))
        rows.sort(key=lambda r: -float(r.get("interactivity_score") or 0))
        args.map_ids = ",".join(str(r["map_id"]) for r in rows[:args.maps])
    print(f"Knob:  {args.knob} ({KNOBS[args.knob]})")
    print(f"Maps:  {len(args.map_ids.split(','))} scenes")

    root = os.path.abspath(os.path.join(args.out, args.knob))
    os.makedirs(root, exist_ok=True)

    summary = []
    for value in args.values:
        value_dir = os.path.join(root, f"{value:g}")
        have = collect_rows(value_dir)
        if not args.aggregate_only and not (args.skip_existing and have):
            run_one(args, value, value_dir)
        rows = collect_rows(value_dir)
        if not rows:
            print(f"  no results for {value}, skipping")
            continue
        metrics = metrics_from_rows(rows)
        collided = sum(
            1 for r in rows
            if parse_percent(r.get("collision_rate", "0")) > 0
        )
        row = {"knob": args.knob, "value": value, "n_maps": len(rows),
               "scenes with collision": collided}
        for key, label in METRICS.items():
            vals = metrics[key]
            row[label] = float(np.mean(vals)) if vals else float("nan")
        summary.append(row)
        print(f"  {args.knob}={value:g}: " + "  ".join(
            f"{k}={row[k]:.2f}" for k in ("collision %", "speed (m/s)", "goal reached %")) +
            f"  scenes_with_collision={collided}/{len(rows)}")

    if not summary:
        sys.exit("No results collected")

    keys = ["knob", "value", "n_maps", "scenes with collision"] + list(METRICS.values())
    with open(os.path.join(root, "results.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)

    make_plot(root, summary, args.knob)
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved -> {root}/results.csv, plot.png, summary.json")


def make_plot(root, summary, knob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["value"] for r in summary]
    collisions = [r["collision %"] for r in summary]
    speeds = [r["speed (m/s)"] for r in summary]
    base = speeds[0] if speeds and speeds[0] else 1.0
    speed_change = [100.0 * (s - base) / base for s in speeds]
    n = summary[0]["n_maps"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].plot(x, collisions, "o-", color="#c0392b")
    axes[0].set_xlabel(f"{knob} ({KNOBS[knob]})")
    axes[0].set_ylabel("collision rate (%)")
    axes[0].set_title("Safety")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, speed_change, "o-", color="#2471a3")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(f"{knob} ({KNOBS[knob]})")
    axes[1].set_ylabel("speed change vs first setting (%)")
    axes[1].set_title("Speed")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{knob}: {n} interactive scenes", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "plot.png"), dpi=160)
    print(f"Plot -> {os.path.join(root, 'plot.png')}")


if __name__ == "__main__":
    main()
