"""Battery health: a visual leakage check on every battery.

A leaking alkaline cell grows a white, crusty, matte deposit (potassium carbonate) at its
ends; an older leak or a corroded contact turns green-blue (copper salts) or rust brown.
Each battery is checked only where leaks appear: the two ends of a cylinder, the face and
rim of a round cell, plus a ring of paper around them (crust often spills onto the paper).

Per pixel, in that zone:
  white crust   bright (as bright as the paper or nearly), almost grey, and ROUGH:
                local RMS of (image - 5x5 median). A median keeps step edges, so the
                battery's own outline does not count as texture; grains and pits do.
  glare         a mirror-like reflection is BRIGHTER than white paper, which a matte crust
                can never be (paper is close to the whitest matte surface). It is found with
                a grey-scale top-hat (small bright details) and is never counted as crust.
  green / rust  colour slicing in HSV, leaving out pixels that have the battery's own body
                colour (a green or copper wrapper is not corrosion).
The pixels are cleaned with closing and opening, tiny pieces are dropped (connected
components), and the remaining area in mm^2 decides: OK, check, or damaged (leaking).
"""

import cv2
import numpy as np
from scipy import ndimage

from . import core

# thresholds chosen by tools/tune_leakage.py on tuning photos only
WHITE_MIN = 0.78      # crust brightness, relative to the paper (paper = 1.0)
GLARE_MIN = 1.06      # brighter than the paper: only a specular reflection can be
CHROMA_WHITE = 0.16   # crust is almost grey
TEX_FLOOR = 0.04      # smallest texture (relative RMS) that can count as rough
SMOOTH_MIN = 0.88    # smooth white deposit inside the outline: at least this bright vs paper
CHROMA_SMOOTH = 0.10  # ... and this grey
HYST = 1.0          # weak crust pixels: at least this share of the texture threshold
                     # (tuning chose 1.0: grow only into equally rough but darker pixels)
TEX_K = 6.0          # texture must be this many robust deviations above the paper's own
BODY_DIST = 0.18      # colour distance (background-divided) below which a pixel is "body colour"
CHECK_MM2 = 1.5       # leak area that asks for a check
DAMAGED_MM2 = 4.0     # leak area that marks the battery as leaking
CHECK_FRAC = 0.006    # the same without a card, as a share of the battery's own area
DAMAGED_FRAC = 0.02

KIND_TEXT = {"white": "white crust (alkaline leakage)", "green": "green-blue corrosion",
             "rust": "rust-brown corrosion"}
ADVICE = {
    "damaged": "Do not use or charge it. Do not touch the crust with bare hands (it is caustic). "
               "Put it in a plastic bag and take it to a battery recycling point. Clean the device's "
               "contacts with a cotton bud dipped in vinegar or lemon juice.",
    "check": "Small deposit or discolouration. Look at it closely; if the crust is real, treat it as leaking.",
    "ok": "No leak seen.",
}


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def pixel_maps(rgb, norm, bg=None):
    """Whole-photo maps used by every battery: brightness and greyness relative to the
    paper, texture, glare and HSV."""
    gray = core.rgb_to_gray(rgb)
    g8 = core.to_u8(gray)
    nmean = norm.mean(axis=-1)
    # texture: local RMS of the BRIGHT part of (image - 7x7 median). A median keeps straight
    # edges, so the battery's outline leaves no residual, while grains of 1-3 px (about
    # 0.3-0.7 mm in a phone photo of A4 paper) are removed by it. Crust is bright grains on
    # darker gaps (a positive residual); embossed text on a cell is dark lines on metal (a
    # negative residual), so only the positive part is kept. It is scaled by the paper's
    # brightness, so noise in a dark corner or on dark metal is not blown up.
    resid = np.maximum(gray - cv2.medianBlur(g8, 7).astype(np.float32), 0)
    rms = np.sqrt(np.maximum(core.box_filter(resid ** 2, 3), 0))
    paper = core.rgb_to_gray(bg) if bg is not None else cv2.GaussianBlur(gray, (0, 0), 4)
    tex = rms / np.maximum(paper, 20.0)
    chroma = np.sqrt(((norm - nmean[..., None]) ** 2).sum(axis=-1)) / np.maximum(nmean, 1e-3)
    # glare: small bright details (top-hat) that are brighter than the paper
    th = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, _disk(7)).astype(np.float32)
    glare = (nmean > GLARE_MIN) & (th > 12)
    glare = cv2.dilate(glare.astype(np.uint8), _disk(2)) > 0
    hsv = core.rgb_to_hsv(rgb.astype(np.float32) / 255.0)
    return {"gray": gray, "tex": tex, "nmean": nmean, "chroma": chroma, "glare": glare,
            "tophat": th, "hsv": hsv, "paper": paper}


def _zone(p, shape):
    """Where a leak would show: the ends of a cylinder (with the paper just beyond them), or a
    round cell's face and a ring round it. Returns (zone, body, window slices)."""
    m = p["mask"]
    H, W = shape
    x0, y0, x1, y1 = p["bbox"]
    width = max(4.0, p["feret_min_px"])
    pad = int(0.6 * width) + 6
    sl = (slice(max(0, y0 - pad), min(H, y1 + pad + 1)), slice(max(0, x0 - pad), min(W, x1 + pad + 1)))
    mw = m[sl].astype(np.uint8)
    ring = int(round(0.3 * width)) + 2
    near = cv2.dilate(mw, _disk(ring)) > 0
    if p.get("type") == "cylinder":
        ys, xs = np.nonzero(mw)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), a = cv2.minAreaRect(pts)
        ang = np.radians(a if w >= h else a + 90)
        u = np.array([np.cos(ang), np.sin(ang)])
        yy, xx = np.mgrid[0:mw.shape[0], 0:mw.shape[1]]
        along = (xx - cx) * u[0] + (yy - cy) * u[1]
        al = along[mw > 0]
        lo, hi = np.percentile(al, 1), np.percentile(al, 99)
        L = hi - lo
        # 22% of the length at each end: wide enough that a shadow joined to the outline (which
        # tilts the fitted axis a little) cannot push the zone past the real tip
        ends = (along > hi - 0.22 * L) | (along < lo + 0.22 * L)
        zone = near & ends
        body = (mw > 0) & (along > lo + 0.3 * L) & (along < hi - 0.3 * L)
    else:
        zone = near
        body = None
    return zone, body, sl


def check_part(p, maps, norm, tex_thr, mm_per_px, others):
    zone, body, sl = _zone(p, maps["gray"].shape)
    mw = p["mask"][sl]
    nm, ch, tx, gl = maps["nmean"][sl], maps["chroma"][sl], maps["tex"][sl], maps["glare"][sl]
    hsv = maps["hsv"][sl]
    nw = norm[sl]
    # the battery's own outline: a thin band where the image steps from body to paper
    edge = cv2.morphologyEx(mw.astype(np.uint8), cv2.MORPH_GRADIENT, _disk(2)) > 0
    # other batteries and the card next to this one belong to them, not to this battery's leak
    other = (others[sl] > 0) & ~mw
    usable = zone & ~gl & ~edge & ~other
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    not_body = np.ones_like(usable)
    if body is not None and body.sum() > 20:
        # body colour by chromaticity (colour / brightness) and hue: both ignore the shading
        # across a round body, so its dark and light sides count as the same colour
        chrom = nw / np.maximum(nw.mean(axis=-1, keepdims=True), 1e-3)
        bchrom = np.median(chrom[body], axis=0)
        bh, bs = float(np.median(h[body])), float(np.median(s[body]))
        dh = np.abs(((h - bh) + 0.5) % 1.0 - 0.5)
        same = np.sqrt(((chrom - bchrom) ** 2).sum(axis=-1)) < BODY_DIST
        if bs > 0.2:
            same |= dh < 0.07
        not_body = ~same

    white = usable & (nm > WHITE_MIN) & (nm < GLARE_MIN) & (ch < CHROMA_WHITE) & (tx > tex_thr)
    green = usable & not_body & (h > 0.22) & (h < 0.55) & (s > 0.22) & (v > 0.25)
    rust = usable & not_body & (h > 0.02) & (h < 0.11) & (s > 0.45) & (v > 0.2) & (v < 0.8)

    def clean(x):
        # opening first removes lone noisy pixels; closing then joins the grains of one deposit
        x = cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, _disk(1))
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, _disk(2))
        return (x > 0) & zone

    # hysteresis (as in Canny): the strong crust pixels above are seeds; they grow, by
    # morphological reconstruction, into touching pixels that are crust-like but less rough.
    # A weak pixel with no seed next to it is never taken, so noise does not grow.
    seeds = clean(white)
    weak = usable & (nm > WHITE_MIN - 0.05) & (nm < GLARE_MIN) & (ch < CHROMA_WHITE) & (tx > HYST * tex_thr)
    if seeds.any():
        seeds = ndimage.binary_propagation(seeds, mask=weak | seeds)
    # smooth white deposit (fresh or wet leak, a paste): not rough, but paper-white and grey
    # INSIDE the battery's own outline at its end. A clean end is metal or wrapper colour,
    # never matte paper-white. An opening with a disc of radius 2 removes thin white print.
    smooth = usable & mw & (nm > SMOOTH_MIN) & (nm < GLARE_MIN) & (ch < CHROMA_SMOOTH)
    smooth = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)) > 0
    raw = {"white": clean(seeds) | (smooth & zone), "green": clean(green), "rust": clean(rust)}
    leak = raw["white"] | raw["green"] | raw["rust"]
    # drop specks: a piece must hold at least 0.4 mm^2 (or 12 px without a card)
    min_px = 12 if not mm_per_px else max(12, int(0.4 / mm_per_px ** 2))
    lab, n = core.label_components(leak)
    if n:
        area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        leak = np.concatenate([[False], area >= min_px])[lab]
    area_px = int(leak.sum())
    kinds = [k for k in ("white", "green", "rust") if (raw[k] & leak).sum() >= max(6, 0.2 * area_px) and area_px]
    frac = area_px / max(int(mw.sum()), 1)
    if mm_per_px:
        area_mm2 = area_px * mm_per_px ** 2
        status = "damaged" if area_mm2 >= DAMAGED_MM2 else "check" if area_mm2 >= CHECK_MM2 else "ok"
    else:
        area_mm2 = None
        status = "damaged" if frac >= DAMAGED_FRAC else "check" if frac >= CHECK_FRAC else "ok"
    if status == "ok":
        reason = "no leak seen"
    else:
        what = " and ".join(KIND_TEXT[k] for k in kinds) or "deposit"
        where = "at an end" if p.get("type") == "cylinder" else "on the face or rim"
        size = f"{area_mm2:.1f} mm^2" if area_mm2 is not None else f"{100 * frac:.1f}% of its area"
        reason = f"{what} {where}, {size}"
    return {"status": status, "area_px": area_px, "area_mm2": None if area_mm2 is None else round(area_mm2, 2),
            "frac": round(frac, 4), "kinds": kinds, "reason": reason,
            "_leak": leak, "_zone": zone, "_sl": sl}


def texture_threshold(maps, parts):
    """Rough means rougher than this photo's own empty paper: median + TEX_K robust deviations."""
    occ = np.zeros(maps["gray"].shape, np.uint8)
    for p in parts:
        occ |= p["mask"].astype(np.uint8)
    free = cv2.dilate(occ, _disk(15)) == 0
    t = maps["tex"][free] if free.sum() > 500 else maps["tex"].ravel()
    med = float(np.median(t))
    mad = float(1.4826 * np.median(np.abs(t - med)))
    return max(TEX_FLOOR, med + TEX_K * mad)


def check(rgb, norm, parts, mm_per_px=None, extra_mask=None, bg=None):
    """Runs the leak check on every part; writes p['health'] and returns whole-photo maps
    for display plus the counts. extra_mask: other objects (the card) to keep out;
    bg: the fitted paper lighting (texture is measured relative to it)."""
    maps = pixel_maps(rgb, norm, bg)
    tex_thr = texture_threshold(maps, parts)
    H, W = rgb.shape[:2]
    others = np.zeros((H, W), np.uint8)
    for p in parts:
        others |= p["mask"].astype(np.uint8)
    if extra_mask is not None:
        others |= extra_mask.astype(np.uint8)
    others = cv2.dilate(others, _disk(2))
    zone_all = np.zeros((H, W), bool)
    leak_all = np.zeros((H, W), bool)
    counts = {"ok": 0, "check": 0, "damaged": 0}
    for p in parts:
        r = check_part(p, maps, norm, tex_thr, mm_per_px, others)
        sl = r.pop("_sl")
        zone_all[sl] |= r.pop("_zone")
        leak_all[sl] |= r.pop("_leak")
        p["health"] = r
        counts[r["status"]] += 1
    return {"maps": maps, "tex_thr": tex_thr, "zone": zone_all, "leak": leak_all, "counts": counts}
