"""Identifier: names each measured battery by nearest-size matching against the catalogue.

For every object:
  shape   cylinder (AA, AAA lying on their side) or button (CR2016, LR44 lying flat), from
          elongation and circularity
  sizes   from the Measurer: cylinder = length and diameter, button = diameter
The object is compared with every catalogue row of its shape. The distance is the Euclidean
norm of the size differences, each divided by its tolerance. Rules that keep it honest:
  unknown    nearest row is outside tolerance (distance > 1): never given a name
  ambiguous  the two nearest rows are nearly equally close (margin): never guessed
"""

import cv2
import numpy as np

# Size tolerances (relative, minimum in mm). The length / button diameter is measured to
# about 1%; the cylinder diameter is a smaller number and is allowed a little more.
TOL = {"size_mm": (0.08, 0.5), "second_mm": (0.15, 0.6)}
AMBIG_MARGIN = 0.35     # second-best minus best distance below this -> ambiguous

ROUND_ELONG = 1.5       # button cells: longest / shortest caliper diameter below this
ROUND_SOLID = 0.85      # ... and solidity (area / convex hull area) above this
LONG_ELONG = 1.8        # cylinders lying on their side are much longer than wide


def part_type(p):
    """Shape class from what the camera sees from above.
      cylinder  elongation > 1.8                                  (AA, AAA on their side)
      button    elongation < 1.5 and solidity > 0.85              (CR2016, LR44 face up)
    A round cell with its shadow beside it looks like a short oval, not a circle, so the
    button test allows some elongation; a cylinder is always far longer than that.
    Holes are ignored: the printing on a cell can leave small false holes in the mask."""
    if p["elongation"] > LONG_ELONG:
        return "cylinder"
    if p["elongation"] < ROUND_ELONG and p["solidity"] > ROUND_SOLID:
        return "button"
    return "other"


def features(p, t):
    """Measured values that correspond to the catalogue columns size_mm and second_mm."""
    if t == "cylinder":
        # robust length and diameter from the two straight sides (median offsets); the
        # caliper diameters are the fallback
        return {"size_mm": p.get("cyl_len_mm", p.get("feret_max_mm")),
                "second_mm": p.get("cyl_d_mm", p.get("feret_min_mm"))}
    if t == "button":
        # minimum caliper width: a shadow falls to one side of the cell and makes it longer
        # in that direction only, so the narrowest width across is still the true diameter
        return {"size_mm": p.get("feret_min_mm"), "second_mm": None}
    return {"size_mm": None, "second_mm": None}


def hu_signature(mask):
    """Log-scaled Hu moments of the filled outline (shape signature, scale/rotation free)."""
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8), binaryImage=True)).ravel()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30))[:4]


def _tol(col, ref):
    return max(TOL[col][0] * ref, TOL[col][1])


def row_distance(f, row):
    """Distance between an object and a catalogue row, in tolerance units."""
    terms = []
    for col in ("size_mm", "second_mm"):
        ref, val = row.get(col), f.get(col)
        if ref is None or val is None:
            continue
        terms.append(((val - ref) / _tol(col, ref)) ** 2)
    return float(np.sqrt(np.sum(terms))) if terms else None


def confidence(d1, d2=None):
    """Confidence of a match, from a Gaussian tolerance model.

    The distance d is in tolerance units, so a measurement error of one tolerance is about
    one standard deviation. Fit = exp(-d1^2 / 2) is the likelihood of the best row relative
    to a perfect match. Separation = 1 - exp(-(d2^2 - d1^2) / 2) is the probability mass
    the runner-up loses against the best (1 when the runner-up is far away). The product
    is high when the object fits one row well and no other row nearly as well:
      d1 = 0.2, far runner-up  -> 0.98      d1 = 0.6, far runner-up -> 0.84
      d1 = 0.2, runner-up 1.0  -> 0.37      (close call, so low)"""
    fit = float(np.exp(-0.5 * d1 ** 2))
    sep = 1.0 if d2 is None else float(1 - np.exp(-0.5 * max(d2 ** 2 - d1 ** 2, 0)))
    return fit * sep


def group_match(p, catalogue, n_max=4):
    """Touching cylinders that could not be split: n cells side by side look like one
    rectangle of the same length and n diameters wide; n cells end to end are n lengths long
    and one diameter wide. The blob is compared with these multiples of every cylinder row,
    with the same tolerance test as a single cell. Returns a match counting n cells, or None."""
    if p["type"] != "cylinder":
        return None
    f = features(p, "cylinder")
    if f["size_mm"] is None or f["second_mm"] is None:
        return None
    best = []
    for i, row in enumerate(catalogue):
        if row["type"] != "cylinder" or row.get("second_mm") is None:
            continue
        L, D = row["size_mm"], row["second_mm"]
        for n in range(2, n_max + 1):
            for how, (gl, gd) in (("side by side", (L, n * D)), ("end to end", (n * L, D))):
                rl, rd = max(gl, gd), min(gl, gd)
                d = float(np.hypot((f["size_mm"] - rl) / _tol("size_mm", rl),
                                   (f["second_mm"] - rd) / _tol("second_mm", rd)))
                best.append((d, i, n, how))
    best.sort()
    if not best or best[0][0] > 1.0:
        return None
    d1, i1, n, how = best[0]
    if len(best) > 1 and best[1][0] - d1 < AMBIG_MARGIN:
        return None
    return {"status": "identified", "row": i1, "name": catalogue[i1]["name"], "count": n,
            "group": f"{n} touching cells, {how}", "distance": d1,
            "confidence": confidence(d1, best[1][0] if len(best) > 1 else None)}


def match(p, catalogue):
    """Nearest-size match with tolerance and ambiguity margin."""
    t = p["type"]
    cands = []
    for i, row in enumerate(catalogue):
        if row["type"] != t:
            continue
        d = row_distance(features(p, t), row)
        if d is not None:
            cands.append((d, i))
    cands.sort()
    res = {"features": features(p, t), "row_type": t,
           "candidates": [(round(d, 3), catalogue[i]["name"]) for d, i in cands[:3]]}
    if not cands:
        return {**res, "status": "unknown", "row": None,
                "reason": "shape is neither a long cylinder nor a round cell"
                if t == "other" else f"no {t} battery in the catalogue"}
    d1, i1 = cands[0]
    if d1 > 1.0:
        g = group_match(p, catalogue)
        if g:
            return {**res, **g}
        return {**res, "status": "unknown", "row": None, "distance": d1,
                "reason": f"nearest catalogue battery ({catalogue[i1]['name']}) is outside tolerance"}
    if len(cands) > 1:
        d2, i2 = cands[1]
        if d2 - d1 < AMBIG_MARGIN:
            return {**res, "status": "ambiguous", "row": None, "distance": d1,
                    "between": [catalogue[i1]["name"], catalogue[i2]["name"]], "reason": "sizes are too close"}
    conf = confidence(d1, cands[1][0] if len(cands) > 1 else None)
    return {**res, "status": "identified", "row": i1, "name": catalogue[i1]["name"],
            "distance": d1, "confidence": conf}


def identify(parts, catalogue, has_scale):
    """Add the shape class and the catalogue match to every object (in place)."""
    for p in parts:
        p["type"] = part_type(p)
        p["hu"] = hu_signature(p["filled"]).tolist()
        if not has_scale:
            p["match"] = {"status": "unknown", "row": None, "candidates": [],
                          "reason": "no credit card found, so sizes in mm are not known"}
        else:
            p["match"] = match(p, catalogue)
    return parts
