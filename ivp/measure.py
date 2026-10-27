"""Scale from the credit card, and per-battery measurement in millimetres."""

import numpy as np
import cv2
from scipy import ndimage

from . import core

CARD_MM = (85.60, 53.98)   # ISO/IEC 7810 ID-1: every bank, credit and ID card
CARD_ASPECT_TOL = 0.35     # 1.24 to 1.94: a tilted camera squashes the card a little


# ----------------------------------------------------------------------------- shape helpers


def outer_contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)


def feret(points):
    """Minimum and maximum Feret (caliper) diameters of a point set, by rotating calipers
    over the convex hull. Returns (min_w, min_angle_deg, max_len, max_angle_deg)."""
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32)).reshape(-1, 2)
    best_min, ang_min = 1e9, 0.0
    best_max, ang_max = 0.0, 0.0
    for t in np.arange(0, 180, 0.5):
        d = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
        proj = hull @ d
        ext = proj.max() - proj.min()
        if ext < best_min:
            best_min, ang_min = ext, t
        if ext > best_max:
            best_max, ang_max = ext, t
    # +1: pixel centres to pixel edges
    return best_min + 1.0, ang_min, best_max + 1.0, ang_max


def holes(mask):
    """Enclosed background regions of one object (4-connected background)."""
    filled = ndimage.binary_fill_holes(mask)
    hl, n = ndimage.label(filled & ~mask)
    areas = ndimage.sum(np.ones_like(hl), hl, index=np.arange(1, n + 1)) if n else np.array([])
    keep = [a for a in areas if a >= 20]
    return filled, len(keep), sorted(keep, reverse=True)


def shape_stats(mask):
    filled, nh, hole_areas = holes(mask)
    c = outer_contour(filled)
    area = float(filled.sum())
    per = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    circ = 4 * np.pi * area / max(per**2, 1)
    solidity = area / max(cv2.contourArea(hull), 1)
    fmin, amin, fmax, amax = feret(c.reshape(-1, 2))
    return {"filled": filled, "area_px": area, "holes": nh, "hole_areas": hole_areas,
            "circularity": float(circ), "solidity": float(solidity), "feret_min": fmin,
            "feret_min_angle": amin, "feret_max": fmax, "feret_max_angle": amax,
            "elongation": fmax / max(fmin, 1), "contour": c}


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def refine_outward(img, contour, inside, reach, frac=0.4, gmin=4.0):
    """Move every outline point outwards along its normal to the FIRST strong edge (scanning
    from the outline outwards, the first local gradient maximum reaching `frac` of the
    strongest one in that scan), refined with a parabola. Used from the coloured wrap of a
    battery: the first edge outside the wrap is the battery's own side, the soft edge of its
    shadow lies further out. Points with no clear edge stay where they are."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return pts
    sm = ndimage.gaussian_filter1d(pts, 2.0, axis=0, mode="wrap")
    tan = np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    probe = pts + 2.5 * nrm
    py = np.clip(np.rint(probe[:, 1]).astype(int), 0, inside.shape[0] - 1)
    px = np.clip(np.rint(probe[:, 0]).astype(int), 0, inside.shape[1] - 1)
    nrm[inside[py, px]] *= -1
    ts = np.arange(-1.0, reach + 1e-9, 0.25)  # from just inside, outwards
    X = pts[:, 0:1] + ts[None] * nrm[:, 0:1]
    Y = pts[:, 1:2] + ts[None] * nrm[:, 1:2]
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[-1])]
    g2 = 0.0
    for ch in chans:
        prof = ndimage.map_coordinates(ch, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
        g2 = g2 + np.gradient(prof, axis=1) ** 2
    g = np.sqrt(g2 / len(chans))
    gmax = g.max(axis=1, keepdims=True)
    is_peak = np.zeros_like(g, dtype=bool)
    is_peak[:, 1:-1] = (g[:, 1:-1] >= g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:])
    ok = is_peak & (g >= np.maximum(frac * gmax, gmin * 0.25))
    found = ok.any(axis=1)
    k = np.clip(np.where(found, ok.argmax(axis=1), 0), 1, len(ts) - 2)
    r = np.arange(len(pts))
    a, b, c = g[r, k - 1], g[r, k], g[r, k + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    t = ts[k] + np.clip(off, -1, 1) * (ts[1] - ts[0])
    t = np.where(found, t, 0.0)
    return pts + t[:, None] * nrm


def wrap_mask(mask, chroma_px):
    """The coloured wrap of a battery: its pixels that are clearly coloured, tidied by a
    closing and an opening, largest piece only. Shadows are grey, so they are never in it."""
    m = (mask & (chroma_px > COLOURED)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def refine_contour(gray, contour, inside, reach=4.0, frac=0.5, gmin=3.0):
    """Sub-pixel edge localisation of an outline. Every contour point is moved along its
    local normal to the outermost strong grey-level edge within +-reach pixels: scanning
    from the outside in, the first local gradient maximum that reaches `frac` of the
    strongest one. (The soft outer edge of a shadow is weaker than the battery's own edge,
    so it is passed over.) The peak is refined with a parabola through three samples.
    `inside` is a boolean image that is True on the object side of the outline."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return pts
    sm = ndimage.gaussian_filter1d(pts, 2.0, axis=0, mode="wrap")
    tan = np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    # orient every normal so that it points out of the object
    probe = pts + 2.5 * nrm
    py = np.clip(np.rint(probe[:, 1]).astype(int), 0, inside.shape[0] - 1)
    px = np.clip(np.rint(probe[:, 0]).astype(int), 0, inside.shape[1] - 1)
    nrm[inside[py, px]] *= -1
    ts = np.arange(reach, -reach - 1e-9, -0.25)  # outside -> inside
    X = pts[:, 0:1] + ts[None] * nrm[:, 0:1]
    Y = pts[:, 1:2] + ts[None] * nrm[:, 1:2]
    chans = [gray] if gray.ndim == 2 else [gray[..., c] for c in range(gray.shape[-1])]
    g2 = 0.0
    for ch in chans:
        prof = ndimage.map_coordinates(ch, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
        g2 = g2 + np.gradient(prof, axis=1) ** 2
    # colour edge strength: Euclidean norm of the per-channel derivatives (scaled to grey units)
    g = np.sqrt(g2 / len(chans))
    gmax = g.max(axis=1, keepdims=True)
    is_peak = np.zeros_like(g, dtype=bool)
    is_peak[:, 1:-1] = (g[:, 1:-1] >= g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:])
    # a peak must also be a real edge (gmin grey levels per pixel), not noise on a faint edge
    ok = is_peak & (g >= np.maximum(frac * gmax, gmin * 0.25))
    found = ok.any(axis=1)
    k = np.where(found, ok.argmax(axis=1), g.argmax(axis=1))
    k = np.clip(k, 1, len(ts) - 2)
    r = np.arange(len(pts))
    a, b, c = g[r, k - 1], g[r, k], g[r, k + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    t = ts[k] + np.clip(off, -1, 1) * (ts[1] - ts[0])
    t = np.where(found, t, 0.0)  # no clear edge: keep the mask boundary
    return pts + t[:, None] * nrm


# ----------------------------------------------------------------------------- scale (card)


def find_reference(labels, rgb, card_mm=CARD_MM):
    """Find the credit card: a rectangle with the card's aspect ratio (85.60 / 53.98 = 1.586)
    that nearly fills its minimum-area rectangle and covers at least 1% of the photo. The
    largest such object is the card. Returns a dict with label id, mm_per_px, its four
    corners and the pixel-to-millimetre homography, or None."""
    best = None
    card_ratio = card_mm[0] / card_mm[1]
    min_card = 0.01 * labels.size

    def card_like(m):
        st = shape_stats(m)
        (_, _), (rw, rh), _ = cv2.minAreaRect(st["contour"])
        L, W = max(rw, rh), min(rw, rh)
        fill = st["area_px"] / max(rw * rh, 1)
        ok = W > 0 and abs(L / W - card_ratio) < CARD_ASPECT_TOL and fill > 0.9
        return ok, st, fill

    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        if m.sum() < min_card:
            continue
        ok, st, fill = card_like(m)
        if not ok:
            # the card may have a thin strip stuck to it (the paper's shadowed edge, a cable,
            # a battery's shadow): an opening with a disc about a fifth of the card's size
            # wipes out anything thin and leaves the card, which is then tested again
            r = max(7, int(0.2 * np.sqrt(m.sum()))) | 1
            op = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
            n, lab, sts, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
            if n < 2:
                continue
            piece = lab == 1 + int(np.argmax(sts[1:, cv2.CC_STAT_AREA]))
            if piece.sum() < min_card:
                continue
            ok, st, fill = card_like(piece)
            if not ok:
                continue
        (rcx, rcy), (rw, rh), rang, corners = refine_rect(rgb, st["filled"])
        L, W = max(rw, rh), min(rw, rh)
        ang = []
        for k in range(4):
            p0, p1, p2 = corners[k - 1], corners[k], corners[(k + 1) % 4]
            v1, v2 = p0 - p1, p2 - p1
            ang.append(float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))))
        cand = {"kind": "card", "label": int(i), "mm_per_px": 0.5 * (card_mm[0] / L + card_mm[1] / W),
                "px": (L, W), "center": (rcx, rcy), "angle": rang, "score": st["area_px"],
                "aspect": L / W, "fill": fill, "corners": corners.tolist(),
                "H": card_homography(corners, (rw, rh), card_mm), "corner_angles": ang}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def refine_rect(rgb, mask):
    """Card rectangle. A coloured card is found in the HSV saturation image, where it is
    strongly coloured and its grey shadow is invisible: the convex hull of the saturated
    pixels inside the card's region (a pale card uses the hull of its mask instead), refined
    to sub-pixel edges. Each side is then fitted as a straight line (robust Huber fit);
    opposite sides give length and width, neighbouring sides meet at the four corners."""
    from .crop import line_intersect
    from .segment import saturation

    sat = ndimage.gaussian_filter(saturation(rgb), 1.0)
    region = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    sm = (sat > np.median(sat) + 0.25) & region
    hull_mask = np.zeros(mask.shape, np.uint8)
    if sm.sum() > 0.6 * mask.sum():
        pts = np.column_stack(np.nonzero(sm))[:, ::-1].astype(np.int32)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(pts), 1)
    else:
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(outer_contour(mask)), 1)
    c = outer_contour(hull_mask)
    (cx, cy), (w0, h0), a0 = cv2.minAreaRect(c)
    gray = core.rgb_to_gray(rgb)
    ref = refine_contour(gray, c, hull_mask > 0, reach=4.0, frac=1.0)
    t = np.radians(a0)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-np.sin(t), np.cos(t)])
    rel = ref - np.array([cx, cy])
    pu, pv = rel @ u, rel @ v
    du, dv = np.abs(pu) / (w0 / 2), np.abs(pv) / (h0 / 2)
    sides = {"+u": (pu > 0) & (du > dv), "-u": (pu < 0) & (du > dv),
             "+v": (pv > 0) & (dv >= du), "-v": (pv < 0) & (dv >= du)}
    off, lines = {}, {}
    for key, sel in sides.items():
        # keep the middle 80% of each side, away from the rounded corners
        along = pv[sel] if "u" in key else pu[sel]
        half = (h0 if "u" in key else w0) / 2
        p = ref[sel][np.abs(along) < 0.8 * half]
        if len(p) < 10:
            rect = cv2.minAreaRect(ref.astype(np.float32))
            return rect[0], rect[1], rect[2], cv2.boxPoints(rect).astype(np.float64)
        vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        off[key] = (np.array([x0, y0]) - np.array([cx, cy])) @ (u if "u" in key else v)
        lines[key] = (vx, vy, x0, y0)
    W = off["+u"] - off["-u"]
    H = off["+v"] - off["-v"]
    corners = []
    for a, b in (("+u", "-v"), ("+u", "+v"), ("-u", "+v"), ("-u", "-v")):
        q = line_intersect(lines[a], lines[b])
        if q is None:
            rect = ((cx, cy), (float(W), float(H)), a0)
            return rect[0], rect[1], rect[2], cv2.boxPoints(rect).astype(np.float64)
        corners.append(q)
    return (cx, cy), (float(W), float(H)), a0, np.array(corners)


def card_homography(corners, size_px, card_mm=CARD_MM):
    """Homography from image pixels to millimetres on the paper plane, from the card's four
    corners (a rectangle of known size). Every outline point mapped through it is in true
    millimetres, which removes any leftover perspective, not only the scale.
    corners are ordered (+u,-v), (+u,+v), (-u,+v), (-u,-v); size_px = (extent along u, along v)."""
    Wu, Hv = size_px
    a, b = (card_mm[0], card_mm[1]) if Wu >= Hv else (card_mm[1], card_mm[0])
    dst = np.float32([[a / 2, -b / 2], [a / 2, b / 2], [-a / 2, b / 2], [-a / 2, -b / 2]])
    return cv2.getPerspectiveTransform(np.float32(corners), dst)


def to_mm(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


# ----------------------------------------------------------------------------- per battery


def cylinder_dims(pts):
    """Robust length and diameter of a cylinder seen from the side (a long rectangle).
    The axis is the long side of the minimum-area rectangle. Each long side is a straight
    edge, so its position is the median across-axis offset of the outline points on it
    (middle 60% of the length): one stray point or a bump of shadow does not move a median,
    while it does move a caliper. The length is between the 1st and 99th percentile along
    the axis, which ignores single outliers at the ends."""
    pts = np.asarray(pts, np.float64)
    (cx, cy), (w, h), a = cv2.minAreaRect(pts.astype(np.float32))
    ang = np.radians(a if w >= h else a + 90)
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-u[1], u[0]])
    rel = pts - np.array([cx, cy])
    al, ac = rel @ u, rel @ v
    lo, hi = np.percentile(al, 1), np.percentile(al, 99)
    mid = (al > lo + 0.2 * (hi - lo)) & (al < hi - 0.2 * (hi - lo))
    top, bot = ac[mid & (ac > 0)], ac[mid & (ac < 0)]
    if len(top) < 5 or len(bot) < 5:
        return float(hi - lo), float(min(w, h))
    return float(hi - lo), float(np.median(top) - np.median(bot))


COLOURED = 0.15      # median chromaticity above which an object counts as coloured


def chromaticity(norm):
    """Colour without brightness: each channel of the background-divided image divided by
    the pixel's mean (r, g, b) / mean, scaled to grey-level units. The paper and every shadow
    on it are grey, so they all become the same flat value; a red or blue battery stands
    out. Edges in this image are battery edges, never shadow edges."""
    mn = norm.mean(axis=-1, keepdims=True)
    return 255.0 * norm / np.maximum(mn, 1e-3) / 3.0


def measure_parts(labels, ref, gray, min_area=150, norm=None):
    """Measure every labelled object (the card is not in `labels`). The outline is refined
    to sub-pixel edges; with a card it is mapped through the card homography into true
    millimetres before the sizes are taken. With `norm` (the background-divided image), a
    coloured object's outline is refined on the chromaticity image instead, where its shadow
    is invisible."""
    k = ref["mm_per_px"] if ref else None
    chrom = chroma_px = None
    if norm is not None:
        chrom = chromaticity(norm)
        mn = norm.mean(axis=-1)
        chroma_px = np.sqrt(((norm - mn[..., None]) ** 2).sum(axis=-1)) / np.maximum(mn, 1e-3)
    parts = []
    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        if m.sum() < min_area:
            continue
        st = shape_stats(m)
        ys, xs = np.nonzero(m)
        # the search reach grows with the object: a contact shadow beside a battery can be
        # several pixels wide, and the true edge (body against shadow) lies inside it
        reach = float(np.clip(0.12 * st["feret_min"], 4.0, 12.0))
        coloured = chroma_px is not None and float(np.median(chroma_px[m])) > COLOURED
        if coloured:
            # the shadow band can be wider than the usual reach; search further inwards
            reach = float(np.clip(0.25 * st["feret_min"], 6.0, 25.0))
            outer = refine_contour(chrom, st["contour"], st["filled"], reach=reach, gmin=6.0)
        else:
            outer = refine_contour(gray, st["contour"], st["filled"], reach=reach)
        fmin, _, fmax, _ = feret(outer.astype(np.float32))
        p = {"label": int(i), "cx": float(xs.mean()), "cy": float(ys.mean()),
             "area_px": st["area_px"], "circularity": st["circularity"], "coloured": coloured,
             "solidity": st["solidity"], "elongation": st["elongation"],
             "feret_min_px": fmin - 1.0, "feret_max_px": fmax - 1.0,   # refined points lie on the edge
             "eq_d_px": 2 * np.sqrt(polygon_area(outer) / np.pi),
             "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
        if ref is not None:
            omm = to_mm(ref["H"], outer)
            a_min, _, a_max, _ = feret(omm.astype(np.float32))
            p["feret_min_mm"], p["feret_max_mm"] = a_min - 1.0, a_max - 1.0   # feret() adds 1
            p["eq_d_mm"] = 2 * np.sqrt(polygon_area(omm) / np.pi)
            p["cyl_len_mm"], p["cyl_d_mm"] = cylinder_dims(omm)
            p["mm_per_px"] = k
            if coloured and st["elongation"] > 1.8:
                # diameter of a coloured cylinder: from its wrap outwards to the first edge,
                # which is the battery's side (its shadow is further out)
                wm = wrap_mask(m, chroma_px)
                if wm is not None and wm.sum() > 0.3 * m.sum():
                    wc = outer_contour(wm)
                    wd = cv2.minAreaRect(wc)[1]
                    side = refine_outward(gray, wc, wm, reach=float(np.clip(0.3 * min(wd), 4.0, 25.0)))
                    p["wrap_d_mm"] = cylinder_dims(to_mm(ref["H"], side))[1]
                    p["cyl_d_mm"] = p["wrap_d_mm"]
        p["mask"] = m
        p["filled"] = st["filled"]
        parts.append(p)
    return parts
