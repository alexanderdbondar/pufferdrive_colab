#!/usr/bin/env python3
"""Build the committed results: tables and figures.

Each dial is evaluated on the scenes that all of its settings share, so the
comparison is paired. The large runs use the 100 most interactive scenes. The
other dials use the 34-scene subset that was already complete.

Usage:
    python final_report.py --runs runs --out results
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import collect_rows, parse_percent  # noqa: E402

SMALL = {
    "alpha_collision": [0, 1, 2, 3],
    "alpha_boundary": [0, 1, 3],
    "goal_speed": [5, 20, 30],
    "alpha_comfort": [0.0, 0.1],
    "alpha_center_bias": [-0.5, 0.5],
    "alpha_l_center": [0.00025, 0.0075],
}
LARGE = {
    "alpha_collision": [0, 1, 3],
    "goal_speed": [5, 30],
}


def load(runs, knob, value):
    rows = collect_rows(os.path.join(runs, knob, f"{value:g}"))
    return {str(r["map_id"]): r for r in rows}


def metrics(rows):
    def pct(key):
        return float(np.mean([parse_percent(r[key]) for r in rows]))

    def num(key):
        return float(np.mean([float(r[key]) for r in rows]))

    return {
        "n_maps": len(rows),
        "collision %": pct("collision_rate"),
        "scenes with collision": sum(
            1 for r in rows if parse_percent(r["collision_rate"]) > 0
        ),
        "offroad %": pct("offroad_rate"),
        "speed (m/s)": num("mean_speed_mps"),
        "goal reached %": 100.0 * float(
            np.mean([str(r["goal_reached"]).strip() == "True" for r in rows])
        ),
        "comfort score": num("score_comfort"),
        "lane-centre score": num("score_l_center"),
    }


def evaluate(runs, knob, values):
    by_value = {v: load(runs, knob, v) for v in values}
    shared = set.intersection(*[set(by_value[v]) for v in values])
    out = []
    for value in values:
        rows = [by_value[value][m] for m in shared]
        row = {"dial": knob, "setting": value}
        row.update(metrics(rows))
        out.append(row)
    return out


def write_table(path, rows):
    keys = ["dial", "setting", "n_maps", "collision %", "scenes with collision",
            "offroad %", "speed (m/s)", "goal reached %", "comfort score",
            "lane-centre score"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_small(path, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes.ravel()

    t = tables["alpha_collision"]
    x = [r["setting"] for r in t]
    ax[0].plot(x, [r["collision %"] for r in t], "o-", color="#c0392b")
    ax[0].set_title("crash aversion")
    ax[0].set_xlabel("alpha_collision")
    ax[0].set_ylabel("collision rate (%)")
    ax[0].set_ylim(bottom=0)

    t = tables["goal_speed"]
    x = [r["setting"] for r in t]
    ax[1].plot(x, [r["speed (m/s)"] for r in t], "o-", color="#2471a3")
    ax[1].set_title("target speed")
    ax[1].set_xlabel("goal_speed")
    ax[1].set_ylabel("mean speed (m/s)")
    ax[1].set_ylim(bottom=0)

    t = tables["alpha_comfort"]
    x = [r["setting"] for r in t]
    ax[2].plot(x, [r["comfort score"] for r in t], "o-", color="#8e44ad")
    ax[2].set_title("comfort weight")
    ax[2].set_xlabel("alpha_comfort")
    ax[2].set_ylabel("comfort score")
    ax[2].set_ylim(0, 1.05)

    for knob, color in (("alpha_center_bias", "#16a085"), ("alpha_l_center", "#d35400")):
        t = tables[knob]
        x = [r["setting"] for r in t]
        ax[3].plot(x, [r["lane-centre score"] for r in t], "o-", color=color, label=knob)
    ax[3].set_title("lane-centre weights")
    ax[3].set_ylabel("lane-centre score")
    ax[3].set_ylim(0, 1.05)
    ax[3].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("34 interactive scenes, one dial at a time", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_large(path, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    t = tables["alpha_collision"]
    x = [r["setting"] for r in t]
    axes[0].plot(x, [r["collision %"] for r in t], "o-", color="#c0392b")
    axes[0].set_title(f"crash aversion ({t[0]['n_maps']} scenes)")
    axes[0].set_xlabel("alpha_collision")
    axes[0].set_ylabel("collision rate (%)")
    axes[0].set_ylim(bottom=0)
    for r in t:
        dx = -18 if r["setting"] == t[-1]["setting"] else 0
        axes[0].annotate(f"{r['scenes with collision']} scenes",
                         (r["setting"], r["collision %"]),
                         textcoords="offset points", xytext=(dx, 9),
                         ha="center", fontsize=8)
    axes[0].set_xlim(-0.3, 3.4)

    t = tables["goal_speed"]
    x = [r["setting"] for r in t]
    axes[1].plot(x, [r["speed (m/s)"] for r in t], "o-", color="#2471a3")
    axes[1].set_title(f"target speed ({t[0]['n_maps']} scenes)")
    axes[1].set_xlabel("goal_speed")
    axes[1].set_ylabel("mean speed (m/s)")
    axes[1].set_ylim(bottom=0)

    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_boundary(path, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["setting"] for r in table]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].plot(x, [r["offroad %"] for r in table], "o-", color="#d35400")
    axes[0].set_title(f"road-edge weight ({table[0]['n_maps']} scenes)")
    axes[0].set_xlabel("alpha_boundary")
    axes[0].set_ylabel("off-road rate (%)")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(x, [r["collision %"] for r in table], "o-", color="#c0392b")
    axes[1].set_title("collisions alongside it")
    axes[1].set_xlabel("alpha_boundary")
    axes[1].set_ylabel("collision rate (%)")
    axes[1].set_ylim(bottom=0)

    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    small_tables = {k: evaluate(args.runs, k, v) for k, v in SMALL.items()}
    large_tables = {k: evaluate(args.runs, k, v) for k, v in LARGE.items()}

    all_small = [r for t in small_tables.values() for r in t]
    all_large = [r for t in large_tables.values() for r in t]
    write_table(os.path.join(args.out, "small_results.csv"), all_small)
    write_table(os.path.join(args.out, "large_results.csv"), all_large)

    plot_small(os.path.join(args.out, "dials_small.png"), small_tables)
    plot_large(os.path.join(args.out, "dials_large.png"), large_tables)
    plot_boundary(os.path.join(args.out, "boundary.png"), small_tables["alpha_boundary"])

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"small": all_small, "large": all_large}, fh, indent=2)

    for name, rows in (("small", all_small), ("large", all_large)):
        print(f"== {name}")
        for r in rows:
            print(f"  {r['dial']:18s} {r['setting']:>8g}  n={r['n_maps']:3d}  "
                  f"coll={r['collision %']:5.2f}%  speed={r['speed (m/s)']:5.2f}  "
                  f"goal={r['goal reached %']:5.1f}%  comfort={r['comfort score']:.3f}  "
                  f"centre={r['lane-centre score']:.3f}")


if __name__ == "__main__":
    main()
