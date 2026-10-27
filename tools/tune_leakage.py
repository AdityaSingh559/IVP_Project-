"""Choose the leakage-check thresholds on TUNING photos only (leak photos 1-5, clean photos
1-2 and the three battery_* test photos). The other photos are kept for tools/test_leakage.py,
so its numbers are measured on photos the thresholds never saw.

The pipeline runs once per photo; only the leak check is repeated for every setting.
Goal: find as many leaks as possible while at most 2% of clean batteries are flagged.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from ivp import health  # noqa: E402
from make_battery_scenes import scene  # noqa: E402
from test_batteries import load  # noqa: E402
from test_leakage import jpeg, owner, run_pipeline, scenes, to_cropped  # noqa: E402

TUNING = {"leak_1_white", "leak_2_green", "leak_3_white", "leak_4_green", "leak_5_white", "clean_1", "clean_2"}


def contexts():
    out = []
    for name, cfg in scenes():
        if name not in TUNING:
            continue
        info = {}
        img, _ = scene(**cfg, info=info)
        c = run_pipeline(jpeg(img))
        leaked = set()
        for lk in info.get("leaks", []):
            p = owner(c, to_cropped(c, lk["photo_xy"]))
            if p is not None:
                leaked.add(p["label"])
        out.append((name, c, leaked, len(info.get("leaks", []))))
        print("prepared", name)
    for f in sorted((ROOT / "data" / "tests").glob("battery_[123]_*.jpg")):
        c = run_pipeline(load(f))
        out.append((f.name, c, set(), 0))
        print("prepared", f.name)
    return out


def score(ctxs):
    hit = n_leak = fa = n_clean = 0
    for _, c, leaked, n in ctxs:
        k = c["ref"]["mm_per_px"] if c["ref"] else None
        health.check(c["rgb"], c["smooth"] / c["bg"], c["parts"], k, c["ref"].get("mask") if c["ref"] else None, c["bg"])
        n_leak += n
        for p in c["parts"]:
            flagged = p["health"]["status"] != "ok"
            if p["label"] in leaked:
                hit += flagged
            else:
                n_clean += 1
                fa += flagged
    return hit / max(n_leak, 1), fa / max(n_clean, 1)


def main():
    ctxs = contexts()
    grid = {"TEX_FLOOR": [0.04, 0.05, 0.06], "WHITE_MIN": [0.78, 0.86],
            "HYST": [0.5, 0.7, 1.0], "CHECK_MM2": [1.5, 2.5]}
    best = None
    for vals in itertools.product(*grid.values()):
        for k, v in zip(grid, vals):
            setattr(health, k, v)
        det, fa = score(ctxs)
        ok = fa <= 0.02
        key = (ok, det, -fa)
        if best is None or key > best[0]:
            best = (key, dict(zip(grid, vals)), det, fa)
        print(dict(zip(grid, vals)), f"detect {100 * det:.0f}%  false alarm {100 * fa:.1f}%")
    print("\nBEST", best[1], f"detect {100 * best[2]:.0f}%  false alarm {100 * best[3]:.1f}%")


if __name__ == "__main__":
    main()
