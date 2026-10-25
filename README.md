# pufferdrive_colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexanderdbondar/pufferdrive_colab/blob/main/pufferdrive_colab.ipynb)

Notebook for training PufferDrive on a Google Colab T4. It covers two things:

1. A setup cell that survives Colab's short VM lifetimes instead of paying a
   ~20 minute build every session.
2. Reward conditioning in my fork of PufferDrive:
   https://github.com/alexanderdbondar/PufferDrive (branch `reward-conditioning`).

## Reward conditioning

In stock PufferDrive every agent in a scene shares the same reward weights. On the
fork branch, driving weights (collision warning, collision penalty, goal radius,
and so on) are drawn from configurable ranges, one value per agent. The policy gets
the weights next to the observation, so one model can express a mix of driving
styles, from cautious to fast.

This follows the conditioning setup in GigaFlow (§F.1). Ranges are set in
`pufferlib/config/ocean/drive.ini`, for example `cond_collision_min`. The notebook
turns them on with `USE_CONDITIONING = True`.

### Checking that it works

A wider range does not prove anything by itself, so the notebook includes a sweep:
freeze a trained policy, replay it with every agent pinned to one allowed value of
a knob (a degenerate `min == max` range), and read the collision rate off. The
example sweep pins the collision penalty at values from 0 down to -1.5. If
conditioning works, the collision rate falls as crash aversion rises. A second pass
moves the goal-radius knob from 2 to 12.

The sweep script is `sweep_conditioning.py` on the fork branch.

## The Colab setup

The setup cell checks for three VM states and only pays for what is missing:

1. Kernel restart or short disconnect: the VM is still alive, so it checks a
   marker and exits in about five seconds.
2. Fresh VM with a warm Drive cache: shallow clone the repo, restore compiled
   extensions and converted maps from Drive, 4 to 6 minutes.
3. First run: clone, build, download and convert the dataset once, 15 to 25
   minutes.

A few choices keep this fast:

1. The repo always comes from git, not from Drive. It has symlinks that Drive
   cannot store, and a fresh shallow clone is quick, so each new VM clones again
   and resets to the branch head.
2. Compiled extensions are cached under a path that includes the git commit hash.
   A new commit rebuilds them instead of loading stale `.so` files.
3. Drive only holds the converted `.bin` maps. The one-time pipeline downloads
   the zip, extracts the JSONs, converts them with `drive.py`, moves the bins into
   the cache, and deletes the JSONs. The cache stays under about 15 GB.
