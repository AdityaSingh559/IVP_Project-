"""Accuracy of the leakage check, measured on drawn photos with known leaks.

Each test photo is a full scene (A4 paper on a table, tilted camera, uneven light, blur, noise,
JPEG) in which some batteries get a white crust or green corrosion of a known size at one end
or on the rim. The whole website pipeline runs on it, and every battery is scored:

  leak found     a leaked battery marked "leaking" (or at least "check")
  false alarm    a clean battery marked "check" or "leaking"
  area error     measured deposit area against the painted area, in mm^2
  count kept     the battery count is still exact (a leak must not break counting)

usage: python tools/test_leakage.py          (writes data/tests/leak_report.json)
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from ivp import catalogue, crop  # noqa: E402
from ivp.steps import STEPS  # noqa: E402
from make_battery_scenes import PRICES, SIZES, scene  # noqa: E402

AREAS = [1.0, 2.0, 4.0, 8.0, 16.0]          # painted deposit sizes, mm^2
NAMES = ["AA battery", "AAA battery", "CR2016 coin cell", "LR44 button cell"]


def jpeg(img_bgr, q=88):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def run_pipeline(rgb):
    cat, _ = catalogue.load(ROOT / "data" / "battery_catalogue.csv")
    c = {"rgb": rgb, "catalogue": cat}
    for st in STEPS:
        st["fn"](c)
    return c


def to_cropped(c, xy):
    """Photo pixel -> pixel in the cropped image the pipeline measured."""
    cm = c.get("crop_map")
    if not cm:
        return xy
    x, y = xy
    if cm["angle"]:
        M = crop.rotate(np.zeros(cm["in_shape"], np.uint8), cm["angle"])[1]
        x, y = M @ np.array([x, y, 1.0])
    return x - cm["offset"][0], y - cm["offset"][1]


def owner(c, xy):
    """The measured battery whose outline is closest to the leak point (within half its width)."""
    x, y = xy
    best, bd = None, 1e9
    for p in c["parts"]:
        ys, xs = np.nonzero(p["mask"])
        d = float(np.min(np.hypot(xs - x, ys - y)))
        if d < bd:
            best, bd = p, d
    if best is not None and bd <= 0.6 * best["feret_min_px"] + 6:
        return best
    return None


def scenes():
    out = []
    k = 0
    for i in range(10):                      # 10 photos with 4 leaks each = 40 leaked batteries
        kind = "white" if i % 2 == 0 else "green"
        leaks = []
        for j, name in enumerate(NAMES):
            leaks.append({"name": name, "kind": kind, "area": AREAS[k % len(AREAS)]})
            k += 1
        out.append((f"leak_{i + 1}_{kind}", dict(seed=300 + i, tilt=0.12 + 0.01 * (i % 5), light=0.3 + 0.03 * (i % 4), leaks=leaks)))
    for i in range(4):                       # clean photos: every battery here is a possible false alarm
        out.append((f"clean_{i + 1}", dict(seed=400 + i, tilt=0.15, light=0.4)))
    return out


DEMO = [{"name": "AA battery", "kind": "white", "area": 16.0}, {"name": "AAA battery", "kind": "green", "area": 12.0},
        {"name": "CR2016 coin cell", "kind": "white", "area": 10.0}, {"name": "LR44 button cell", "kind": "green", "area": 8.0}]


def make_demo():
    """The test photo offered on the website: 4 clearly leaking batteries (not used in the numbers)."""
    info = {}
    img, truth = scene(seed=510, tilt=0.12, light=0.35, leaks=DEMO, info=info)
    cv2.imwrite(str(ROOT / "data" / "tests" / "battery_4_leaking.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    tf = ROOT / "data" / "tests" / "battery_truth.json"
    t = json.loads(tf.read_text())
    cnt = {n: truth.count(n) for n in SIZES}
    t["battery_4_leaking.jpg"] = {"counts": cnt, "total": sum(PRICES[n] * v for n, v in cnt.items()),
                                  "leaking": [lk["name"] for lk in info["leaks"]]}
    tf.write_text(json.dumps(t, indent=1))


def main():
    if "--demo" in sys.argv:
        make_demo()
        return
    rows, per_photo = [], []
    for name, cfg in scenes():
        info = {}
        img, truth = scene(**cfg, info=info)
        rgb = jpeg(img)
        c = run_pipeline(rgb)
        # counting still right?
        counts = {}
        for p in c["parts"]:
            m = p["match"]
            if m["status"] == "identified":
                counts[m["name"]] = counts.get(m["name"], 0) + m.get("count", 1)
        exact = all(counts.get(n, 0) == truth.count(n) for n in SIZES) and sum(counts.values()) == len(truth)
        leaked_ids = {}
        for lk in info.get("leaks", []):
            p = owner(c, to_cropped(c, lk["photo_xy"]))
            if p is not None:
                leaked_ids[p["label"]] = lk
            else:
                rows.append({"photo": name, "battery": lk["name"], "leaked": True, "kind": lk["kind"],
                             "true_mm2": lk["true_area"], "status": "missed (battery not found)", "measured_mm2": None})
        for p in c["parts"]:
            h = p.get("health", {})
            lk = leaked_ids.get(p["label"])
            nm = p["match"].get("name") or p["match"]["status"]
            rows.append({"photo": name, "battery": lk["name"] if lk else nm, "leaked": lk is not None,
                         "kind": lk["kind"] if lk else None, "true_mm2": lk["true_area"] if lk else 0.0,
                         "status": h.get("status"), "measured_mm2": h.get("area_mm2")})
        flagged = sum(1 for p in c["parts"] if p.get("health", {}).get("status") != "ok")
        per_photo.append({"photo": name, "count_exact": bool(exact), "leaks": len(info.get("leaks", [])), "flagged": flagged})
        print(f"{name:16s} count {'EXACT' if exact else 'WRONG'}  leaks {len(info.get('leaks', []))}  flagged {flagged}  "
              f"health {c.get('health', {}).get('counts')}")

    from tune_leakage import TUNING
    held = [r for r in rows if r["photo"] not in TUNING]
    print("\n=== HELD-OUT photos (never used to choose the thresholds) ===")
    rep = summary(held, [p for p in per_photo if p["photo"] not in TUNING])
    print("\n=== ALL photos ===")
    rep_all = summary(rows, per_photo)
    rep["all_photos"] = {k: rep_all[k] for k in ("detection_rate", "false_alarm_rate", "precision", "leaked_batteries", "clean_batteries")}
    rep["photos_detail"] = per_photo
    (ROOT / "data" / "tests" / "leak_report.json").write_text(json.dumps(rep, indent=1))


def summary(rows, per_photo):
    leaked = [r for r in rows if r["leaked"]]
    clean = [r for r in rows if not r["leaked"]]
    found_any = sum(r["status"] in ("check", "damaged") for r in leaked)
    found_dmg = sum(r["status"] == "damaged" for r in leaked)
    false_any = sum(r["status"] in ("check", "damaged") for r in clean)
    false_dmg = sum(r["status"] == "damaged" for r in clean)
    print("\nconfusion matrix (rows = truth, columns = result)")
    print(f"{'':12s}{'ok':>8s}{'check':>8s}{'leaking':>9s}")
    for lbl, grp in (("leaked", leaked), ("clean", clean)):
        print(f"{lbl:12s}" + "".join(f"{sum(r['status'] == s for r in grp):>{w}d}" for s, w in (("ok", 8), ("check", 8), ("damaged", 9))))
    by_area = []
    print("\ndetection by deposit size")
    for a in AREAS:
        g = [r for r in leaked if abs(r["true_mm2"] - a) / a < 0.5]
        if not g:
            continue
        fa = sum(r["status"] in ("check", "damaged") for r in g) / len(g)
        fd = sum(r["status"] == "damaged" for r in g) / len(g)
        errs = [abs(r["measured_mm2"] - r["true_mm2"]) for r in g if r["measured_mm2"] is not None and r["status"] != "ok"]
        by_area.append({"area_mm2": a, "n": len(g), "flagged": round(fa, 2), "leaking": round(fd, 2),
                        "mean_abs_area_error_mm2": round(float(np.mean(errs)), 2) if errs else None})
        print(f"  {a:5.1f} mm^2: n={len(g):2d}  flagged {100 * fa:5.1f}%  marked leaking {100 * fd:5.1f}%"
              + (f"  area error {np.mean(errs):.2f} mm^2" if errs else ""))
    by_kind = {}
    for kind in ("white", "green"):
        g = [r for r in leaked if r["kind"] == kind]
        by_kind[kind] = round(sum(r["status"] in ("check", "damaged") for r in g) / max(len(g), 1), 2)
    rep = {
        "photos": len(per_photo), "leaked_batteries": len(leaked), "clean_batteries": len(clean),
        "detection_rate": round(found_any / max(len(leaked), 1), 3),
        "leaking_rate": round(found_dmg / max(len(leaked), 1), 3),
        "false_alarm_rate": round(false_any / max(len(clean), 1), 3),
        "false_leaking_rate": round(false_dmg / max(len(clean), 1), 3),
        "precision": round(found_any / max(found_any + false_any, 1), 3),
        "counts_exact": sum(p["count_exact"] for p in per_photo),
        "by_area": by_area, "by_kind": by_kind,
        "confusion": {lbl: {s: sum(r["status"] == s for r in grp) for s in ("ok", "check", "damaged")}
                      for lbl, grp in (("leaked", leaked), ("clean", clean))},
    }
    print(f"\nleaked batteries found: {found_any}/{len(leaked)} ({100 * rep['detection_rate']:.0f}%), "
          f"marked leaking {found_dmg}/{len(leaked)}")
    print(f"false alarms on clean batteries: {false_any}/{len(clean)} ({100 * rep['false_alarm_rate']:.1f}%)")
    print(f"precision {100 * rep['precision']:.0f}%   counts still exact in {rep['counts_exact']}/{len(per_photo)} photos")
    print(f"by kind: {by_kind}")
    return rep


if __name__ == "__main__":
    main()
