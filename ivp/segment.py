"""Segmentation: finds the batteries and the card on the white paper.

Steps (each returns its intermediate image so the website can show before / after):
  background model  -> background division (evens out lighting and vignetting)
  normalised image  -> Euclidean colour distance from the background, Otsu threshold
  shadow test (HSV) -> pixels that are only a darker copy of the paper are dropped
  region filling    -> enclosed gaps (printing, glare on metal) are filled, real gaps that
                       look like the paper are kept
  morphology        -> opening / closing to remove specks and close gaps
  splitting         -> distance transform + h-maxima markers + watershed
"""

import numpy as np
import cv2
from scipy import ndimage

from . import core


# ----------------------------------------------------------------------------- lighting


def saturation(rgb):
    """HSV saturation S = (max - min) / max of an RGB image (0..1)."""
    rgb = rgb.astype(np.float32)
    mx, mn = rgb.max(-1), rgb.min(-1)
    return (mx - mn) / np.maximum(mx, 1)


def coloured_objects(rgb, min_area=2500):
    """Colour slicing: large strongly coloured objects (the card, coloured battery bodies)
    found by their HSV saturation compared with the paper. Pieces that lie
    close together (the two blue bands of a card around its white stripe) are grouped and
    replaced by their convex hull."""
    s = saturation(rgb)
    s_bg = np.median(s)
    m = (s > s_bg + 0.25).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    groups = cv2.dilate(m, np.ones((41, 41), np.uint8))
    lab, n = ndimage.label(groups)
    out = np.zeros(m.shape, np.uint8)
    for i in range(1, n + 1):
        pts = np.column_stack(np.nonzero((lab == i) & (m > 0)))[:, ::-1]
        if len(pts) < min_area:
            continue
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    return cv2.dilate(out, np.ones((9, 9), np.uint8)) > 0


def fit_background(rgb, step=8, sigma=4.0, iters=6):
    """Lighting of the empty paper, estimated by robust normalised convolution.

    On a coarse grid the image is smoothed with a Gaussian that only averages pixels currently
    believed to be background (weight w): bg = G*(I w) / G*(w). Pixels that differ from that
    estimate (batteries, shadows, the card) get weight 0 and the estimate is repeated. The
    result follows vignetting and dark corners but not the objects themselves."""
    h, w, _ = rgb.shape
    small = cv2.resize(rgb.astype(np.float32), (w // step, h // step), interpolation=cv2.INTER_AREA)
    excl = cv2.resize(coloured_objects(rgb).astype(np.uint8), (w // step, h // step),
                      interpolation=cv2.INTER_NEAREST) > 0
    # anything much darker than the paper is an object, never background: a large black card
    # would otherwise be blurred into the lighting estimate and then vanish after division
    bright = small.mean(axis=-1)
    excl |= bright < 0.6 * np.percentile(bright, 90)
    wgt = (~excl).astype(np.float32)
    est = small.copy()
    for _ in range(iters):
        num = np.stack([ndimage.gaussian_filter(small[..., c] * wgt, sigma) for c in range(3)], -1)
        den = ndimage.gaussian_filter(wgt, sigma)[..., None]
        wide_num = np.stack([ndimage.gaussian_filter(small[..., c] * wgt, sigma * 4) for c in range(3)], -1)
        wide_den = ndimage.gaussian_filter(wgt, sigma * 4)[..., None]
        # where few background pixels are nearby (under a large card) blend smoothly towards
        # a wider average, so the estimate has no steps
        alpha = np.clip(den / 0.3, 0, 1)
        est = alpha * (num / np.maximum(den, 1e-6)) + (1 - alpha) * (wide_num / np.maximum(wide_den, 1e-6))
        # relative deviation, with a brightness floor so dark corners are not over-sensitive
        dev = np.sqrt((((small - est) / np.maximum(est, 80)) ** 2).sum(-1))
        wgt = ((dev < 0.07) & ~excl).astype(np.float32)
    bg = cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 4.0)


# ----------------------------------------------------------------------------- object mask


def object_mask(rgb, bg):
    """Return dict with the normalised image, colour distance, shadow map and the raw mask."""
    n = rgb.astype(np.float32) / bg  # 1.0 on the background
    # Euclidean distance from the background colour; the brightness floor keeps noise in a
    # very dark corner from being amplified by the division
    dist = np.sqrt((((rgb - bg) / np.maximum(bg, 70)) ** 2).sum(axis=-1))
    d8 = core.to_u8(np.clip(dist, 0, 1) * 255)
    otsu = core.otsu_threshold(d8)
    # Otsu is pulled up by very different objects (the card, black cells); the noise floor of the
    # background gives a second bound, and the lower of the two is used.
    bgpix = dist[dist < np.percentile(dist, 60)]
    noise = 1.4826 * np.median(np.abs(bgpix - np.median(bgpix)))
    noise_t = 255 * max(0.10, np.median(bgpix) + 8 * noise)
    t = int(np.clip(min(otsu, noise_t), 18, 90))

    # Shadow test in HSV terms: a shadow keeps the hue and saturation of the background (its
    # normalised colour is grey, n_r ~ n_g ~ n_b) and is only somewhat lower in value. A
    # battery is either coloured or clearly darker or brighter than a soft shadow.
    # Shadows are smooth, so the test runs on a smoothed copy where noise is low.
    # A pixel must also pass a looser test on its own, so dark part edges are not taken.
    def grey_test(img):
        mn = img.mean(axis=-1)
        ch = np.sqrt(((img - mn[..., None]) ** 2).sum(axis=-1)) / np.maximum(mn, 1e-3)
        return mn, ch

    # 5x5 median per channel: removes noise like a blur but does not spread a part's colour
    # into the shadow next to it
    nm = np.stack([ndimage.median_filter(n[..., c], size=5) for c in range(3)], -1)
    mean_n, chroma = grey_test(nm)
    mean_p, chroma_p = grey_test(n)
    # relative noise grows where the tray is dark, so the colour tolerance grows with it
    tol = np.clip(130.0 / np.maximum(bg.mean(axis=-1), 1), 1.0, 3.0)
    # ... and with the colour noise measured on this photo's own empty tray
    empty = dist < np.percentile(dist, 50)
    c_med = max(0.02, 4.0 * float(np.median(chroma[empty])))
    c_pix = max(0.045, 4.0 * float(np.median(chroma_p[empty])))
    shadow = ((chroma < c_med * tol) & (mean_n > 0.74) & (mean_n < 0.98)
              & (chroma_p < c_pix * tol) & (mean_p > 0.74))

    raw = (d8 > t) & ~shadow
    return {"norm": n, "dist": dist, "dist_u8": d8, "otsu": otsu, "thresh": t, "shadow": shadow,
            "raw": raw, "chroma": chroma, "mean_n": mean_n, "shadow_tol": (c_med, c_pix)}


def fill_regions(mask, dist, thresh):
    """Fill enclosed regions of the mask that do not look like the paper (printing, glare on
    a metal cap); keep those that do as holes. Enclosed regions are the 4-connected components of the background
    that do not touch the image border. A region is a hole when its median colour distance
    from the background is below the same threshold that defines an object pixel."""
    inv = ~mask
    lab, n = ndimage.label(inv)  # 4-connectivity for background (dual of 8 for objects)
    if n == 0:
        return mask.copy()
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    fill = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if i in border or sl is None:
            continue
        # tiny pockets are always filled; larger ones only when they are not background-like
        fill[i] = area[i - 1] < 30 or np.median(dist[sl][lab[sl] == i]) > thresh
    return mask | fill[lab]


def tidy(mask, k_open=5, k_close=5, min_area=150):
    """Opening removes specks, closing bridges one-pixel gaps, tiny components are dropped."""
    m = mask.astype(np.uint8)
    m = core.closing(m, k_close)
    m = core.opening(m, k_open)
    lab, n = core.label_components(m > 0)
    if n == 0:
        return m > 0
    area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    ok = np.concatenate([[False], area >= min_area])
    return ok[lab]


def _on_border(m):
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def _pieces(m, k, min_area):
    """Parts of m that survive an opening with a k x k disk (everything thinner than k goes).
    Works on the bounding box of m, padded, so large disks stay fast."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.pad(m[y0:y1, x0:x1], k).astype(np.uint8)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    op = cv2.morphologyEx(crop, cv2.MORPH_OPEN, disk)[k:-k, k:-k] > 0
    pl, pn = ndimage.label(op, structure=np.ones((3, 3)))
    out = []
    for j in range(1, pn + 1):
        pj = pl == j
        if pj.sum() >= min_area:
            full = np.zeros(m.shape, bool)
            full[y0:y1, x0:x1] = pj
            out.append(full)
    return out


def _is_round(m):
    from . import measure

    st = measure.shape_stats(m)
    return st["holes"] == 0 and st["circularity"] > 0.8 and st["elongation"] < 1.15


def _is_card(m):
    """A card-shaped piece: fills its minimum-area rectangle, with the card's aspect ratio."""
    from . import measure

    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return False
    c = max(cs, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    if min(w, h) < 1:
        return False
    ratio = measure.CARD_MM[0] / measure.CARD_MM[1]
    return (abs(max(w, h) / min(w, h) - ratio) < measure.CARD_ASPECT_TOL and m.sum() / (w * h) > 0.9
            and m.sum() > 0.01 * m.size)


def detach(mask, min_area=1500):
    """Clean up two things a real photo adds, leaving every other blob exactly as it was:
    1. Necks: a blob is opened with growing disks (15 to 181 px); if that frees a round solid
       piece or a card-shaped piece (a cell or the card joined to a battery by a shadow), the
       blob is cut by marker-controlled
       watershed on the distance transform, with the freed pieces as markers, so each piece
       gets back its full outline and the cut runs along the narrowest place.
    2. Edge strips: a blob touching the photo border is opened with a 31 px disk; what does not
       survive and lies more than 16 px from a surviving body (the edge of a folder or a
       table) is dropped, and surviving pieces that are long thin strips along the border are
       dropped too."""
    from . import measure

    out = mask.copy()
    notes = []
    lab, n = core.label_components(out)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < 2 * min_area:
            continue
        for k in (15, 25, 35, 51, 71, 91, 121, 151, 181):
            pieces = _pieces(c, k, min_area)
            if len(pieces) >= 2 and any(_is_round(p) or _is_card(p) for p in pieces):
                ys, xs = np.nonzero(c)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                cc = c[y0:y1, x0:x1]
                dt = cv2.distanceTransform(cc.astype(np.uint8), cv2.DIST_L2, 5)
                markers = np.zeros(cc.shape, np.int32)
                for j, p in enumerate(pieces, start=1):
                    markers[p[y0:y1, x0:x1]] = j
                lab2 = watershed_flood(-dt, markers, cc).astype(np.float32)
                # a one-pixel gap between the pieces keeps them separate components
                gap = (cv2.dilate(lab2, np.ones((3, 3))) != lab2) & cc
                out[y0:y1, x0:x1][gap] = False
                notes.append(f"cut a blob into {len(pieces)} pieces (the join was narrower than {k} px)")
                break
    lab, n = core.label_components(out)
    for i in range(1, n + 1):
        c = lab == i
        if not _on_border(c):
            continue
        keep = np.zeros_like(c)
        for p in _pieces(c, 31, min_area):
            if _on_border(p):
                st = measure.shape_stats(p)
                if st["elongation"] > 5 or st["solidity"] < 0.5:
                    continue
            # keep the blob's own pixels near this body (restores what the opening removed)
            keep |= (cv2.dilate(p.astype(np.uint8), np.ones((33, 33), np.uint8)) > 0) & c
        removed = int((c & ~keep).sum())
        if removed:
            out &= ~(c & ~keep)
            notes.append(f"removed {removed} px of strip along the photo edge")
    return out, notes


def boundary(mask):
    """Morphological boundary extraction: A minus (A eroded by B)."""
    m = mask.astype(np.uint8)
    return (m - core.erode(m, 3)) > 0


# ----------------------------------------------------------------------------- splitting


def h_maxima(f, h):
    """Regional maxima of f - h reconstructed under f (morphological reconstruction by
    dilation). Removes maxima whose height above their surroundings is below h."""
    rec = reconstruct(f - h, f)
    # markers are the regional maxima of the reconstruction: one flat top per dome, so a long
    # ridge of almost equal values (the middle of an AA cell) is one marker, not
    # a row of separate pixels
    top = (rec - reconstruct(rec - 0.01, rec)) > 0.005
    # two equal peaks a pixel or two apart (common in a distance map) form one marker
    return cv2.dilate(top.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def reconstruct(seed, mask):
    """Morphological reconstruction by dilation of seed under mask (seed <= mask)."""
    prev = None
    rec = np.minimum(seed, mask).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    while prev is None or not np.array_equal(prev, rec):
        prev = rec
        rec = np.minimum(cv2.dilate(rec, kernel), mask)
    return rec


def watershed_flood(elev, markers, region):
    """Marker-controlled watershed by priority flooding (Meyer): pixels of `region` are
    claimed by the marker that reaches them first when flooding from low to high `elev`."""
    import heapq

    lab = markers.copy()
    h, w = elev.shape
    heap = []
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys, xs):
        heapq.heappush(heap, (float(elev[y, x]), int(y), int(x)))
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        _, y, x = heapq.heappop(heap)
        l = lab[y, x]
        for dy, dx in nbrs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and lab[yy, xx] == 0 and region[yy, xx]:
                lab[yy, xx] = l
                heapq.heappush(heap, (float(elev[yy, xx]), yy, xx))
    return lab


NECK = 0.85     # a watershed cut longer than this share of the thinner piece's width is undone


def merge_wide_cuts(lab, dist):
    """Undo watershed cuts that do not go through a neck. Two objects that touch meet in a
    narrow neck, so the cut between them is short compared with their widths; one object
    split by mistake (a battery with a ragged shadow) is cut right across its full width.
    For every pair of neighbouring pieces the cut length is compared with the thinner
    piece's width (twice its largest distance-transform value); wide cuts are merged."""
    lab = lab.copy()
    changed = True
    while changed:
        changed = False
        ids = [i for i in np.unique(lab) if i]
        if len(ids) < 2:
            break
        width = {i: 2.0 * float(dist[lab == i].max()) for i in ids}
        for a in ids:
            grown = cv2.dilate((lab == a).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            for b in np.unique(lab[grown & (lab != a) & (lab > 0)]):
                cut = int((grown & (lab == b)).sum())
                if cut > NECK * min(width[a], width[b]):
                    lab[lab == b] = a
                    changed = True
                    break
            if changed:
                break
    return lab


def split_touching(mask, h=3.0):
    """Distance transform of the hole-filled mask, h-maxima as markers, watershed on the
    negated distance. Blobs with one marker are kept whole; only blobs with two or more
    markers are flooded. Returns a label image (0 = background), the distance map, markers."""
    solid = ndimage.binary_fill_holes(mask)
    dt = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    comp, nc = core.label_components(solid)
    peaks = np.zeros(mask.shape, bool)
    lab = np.zeros(mask.shape, np.int32)
    next_id = 1
    objs = ndimage.find_objects(comp)
    for i, sl in enumerate(objs, start=1):
        c = comp[sl] == i
        d = np.where(c, dt[sl], 0)
        # dome height needed for a separate object scales with the blob's thickness, so ragged
        # edges on a thin cell do not create extra markers
        hh = float(np.clip(0.3 * d.max(), h, 2 * h))
        pk = h_maxima(np.pad(d, 1), hh)[1:-1, 1:-1] & c & (d > 4)
        peaks[sl] |= pk
        mk, _ = ndimage.label(pk, structure=np.ones((3, 3)))
        ids = np.unique(mk[mk > 0])
        if len(ids) <= 1:
            lab[sl][c] = next_id
            next_id += 1
            continue
        remap = np.zeros(mk.max() + 1, np.int32)
        remap[ids] = np.arange(next_id, next_id + len(ids))
        flooded = watershed_flood(-dt[sl], remap[mk], c)
        flooded = merge_wide_cuts(np.where(c, flooded, 0), d)
        lab[sl][c] = flooded[c]
        next_id += len(ids)
    lab[~mask] = 0
    return lab, dt, peaks


# ----------------------------------------------------------------------------- pipeline


def denoise(rgb):
    """Median 3x3 per channel (removes JPEG and sensor specks) then a light Gaussian."""
    med = np.stack([core.median3x3(rgb[..., c].astype(np.float32)) for c in range(3)], -1)
    return core.gaussian_blur(med, 1.0)
