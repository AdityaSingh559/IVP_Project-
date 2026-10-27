"""Run the website's pipeline steps on the battery test scenes and compare with the truth.

usage: python tools/test_batteries.py [image ...]   (default: data/tests/battery_*.jpg)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp import catalogue  # noqa: E402
from ivp.steps import STEPS  # noqa: E402


def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > 1600:
        k = 1600 / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return np.asarray(im).copy()


def run(path, quiet=False):
    cat, _ = catalogue.load(ROOT / "data" / "battery_catalogue.csv")
    c = {"rgb": load(path), "catalogue": cat}
    for st in STEPS:
        r = st["fn"](c)
        if not quiet and st["id"] in ("crop", "scale", "split", "measure", "identify"):
            print(f"  [{st['id']}] {r['status']}: {r['summary']}")
    counts = {}
    confs = []
    for p in c["parts"]:
        m = p["match"]
        if m["status"] == "identified":
            counts[m["name"]] = counts.get(m["name"], 0) + m.get("count", 1)
            confs.append(m["confidence"])
        else:
            counts["(" + m["status"] + ")"] = counts.get("(" + m["status"] + ")", 0) + 1
    return c, counts, confs


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "data" / "tests").glob("battery_*.jpg"))
    truth_f = ROOT / "data" / "tests" / "battery_truth.json"
    truth = json.loads(truth_f.read_text()) if truth_f.exists() else {}
    exact = 0
    for p in paths:
        print(p.name)
        c, counts, confs = run(p)
        tot = c["invoice"]["grand_total"]
        t = truth.get(p.name)
        line = f"  counts {counts}  total {tot}  mean confidence {np.mean(confs) * 100:.0f}% (min {min(confs) * 100:.0f}%)" if confs else f"  counts {counts}"
        print(line)
        if t:
            ok = all(counts.get(k, 0) == v for k, v in t["counts"].items()) and len(counts) == len(t["counts"])
            exact += ok
            print(f"  truth  {t['counts']}  total {t['total']}  ->  {'EXACT' if ok else 'WRONG'}")
        for part in sorted(c["parts"], key=lambda q: q["no"]):
            m = part["match"]
            f = m.get("features") or {}
            print(f"    #{part['no']:2d} {part['type']:8s} size {f.get('size_mm') or 0:6.2f} second {f.get('second_mm') or 0:6.2f} "
                  f"-> {m['status']:10s} {m.get('name') or m.get('reason')}  conf {m.get('confidence') or 0:.2f}")
    if truth:
        print(f"\nexact inventories: {exact} of {len(paths)}")


if __name__ == "__main__":
    main()
