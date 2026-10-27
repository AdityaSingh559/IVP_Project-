"""Second try for touching batteries that the watershed could not separate.

Runs only on objects that could not be named. Two ways of cutting an object are tried:
  shrink and grow back  the object is eroded a little at a time until it falls apart into
                        pieces (a battery's end pressed against another battery's side is
                        a thin join), then every piece is grown back inside the original
                        outline, so each battery keeps its full size
  corner cuts           every battery is a straight rectangle or a circle, so where two meet
                        the outline has a deep inward corner (a convexity defect); a cut
                        between two such corners follows the line where they touch
A split is accepted only if every piece is then named from the catalogue on its own
(each piece may itself be split again, up to a few levels). If no split names every
piece, the object is left exactly as it was. Objects that were already named are never
touched, so this can only turn an unknown into batteries, never change a named battery.
"""
import itertools

import cv2
import numpy as np

from . import identify, measure

MAX_DEPTH = 3          # a blob of up to 4 batteries can be taken apart
BUDGET = 160           # most piece measurements per object (keeps it fast)
SHRINK_MAX = 0.4       # erode by at most this share of the thinnest battery's diameter
DEFECT_MIN = 0.15      # an inward corner counts when at least this deep (share of thinnest diameter)
MARGIN = 30            # pixels of context around an object when it is measured
CAP = 0.3              # end caps reach this far past the wrapper (share of thinnest diameter, ~3 mm)


def _disc(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


class _Ctx:
    def __init__(self, ref, gray, norm, catalogue):
        self.ref, self.gray, self.norm, self.cat = ref, gray, norm, catalogue
        px = 1.0 / ref["mm_per_px"]
        diam = [r["second_mm"] or r["size_mm"] for r in catalogue]
        self.d_min = min(diam) * px                       # thinnest battery, in px
        self.d_max = max(diam) * px                       # widest battery, in px
        self.min_piece = (0.5 * self.d_min) ** 2          # smaller pieces are crumbs
        self.calls = 0

    def measure(self, mask):
        """Measure one piece exactly as the Measure step does, in a window around it."""
        self.calls += 1
        ys, xs = np.nonzero(mask)
        H, W = mask.shape
        x0, y0 = max(0, xs.min() - MARGIN), max(0, ys.min() - MARGIN)
        x1, y1 = min(W, xs.max() + MARGIN + 1), min(H, ys.max() + MARGIN + 1)
        T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
        ref = dict(self.ref, H=np.asarray(self.ref["H"], np.float64) @ T)
        lab = mask[y0:y1, x0:x1].astype(np.int32)
        norm = self.norm[y0:y1, x0:x1] if self.norm is not None else None
        got = measure.measure_parts(lab, ref, self.gray[y0:y1, x0:x1], norm=norm)
        if not got:
            return None
        p = got[0]
        p["cx"] += x0
        p["cy"] += y0
        b = p["bbox"]
        p["bbox"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
        for key in ("mask", "filled"):
            full = np.zeros(mask.shape, bool)
            full[y0:y1, x0:x1] = p[key]
            p[key] = full
        p["type"] = identify.part_type(p)
        p["hu"] = identify.hu_signature(p["filled"]).tolist()
        p["match"] = identify.match(p, self.cat)
        return p


def _named(p):
    """Named as one single battery (a group count is not accepted for a piece)."""
    m = p["match"] if p else None
    return m is not None and m["status"] == "identified" and m.get("count", 1) == 1


def _pieces(lab, region):
    out = []
    for i in np.unique(lab):
        if i == 0:
            continue
        pm = (lab == i) & region
        out.append(pm)
    return out


def shrink_splits(mask, ctx):
    """Erode until the object falls apart; every new number of pieces is one candidate.
    The pieces are grown back to fill the original outline (each pixel goes to the
    nearest piece), so nothing of the object is lost."""
    found, seen = [], set()
    m8 = mask.astype(np.uint8)
    rmax = SHRINK_MAX * ctx.d_min
    r = 1.0
    while r <= rmax:
        er = cv2.erode(m8, _disc(r))
        n, lab, st, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.25 * ctx.min_piece]
        if len(keep) >= 2 and len(keep) not in seen:
            seen.add(len(keep))
            seeds = np.zeros(mask.shape, np.int32)
            for k, i in enumerate(keep, start=1):
                seeds[lab == i] = k
            # nearest seed for every pixel (distance transform with labels), inside the outline
            _, near = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
            # map the transform's own labels back to seed numbers
            lut = np.zeros(near.max() + 1, np.int32)
            lut[near[seeds > 0]] = seeds[seeds > 0]
            grown = np.where(mask, lut[near], 0)
            pcs = _pieces(grown, mask)
            if all(pc.sum() >= ctx.min_piece for pc in pcs):
                found.append(("shrink", pcs))
        r += max(1.0, 0.05 * ctx.d_min)
    return found


def _grow(seeds, mask):
    """Every pixel of the outline goes to its nearest seed, so no part of the object is lost."""
    _, near = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(near.max() + 1, np.int32)
    lut[near[seeds > 0]] = seeds[seeds > 0]
    return _pieces(np.where(mask, lut[near], 0), mask)


def _cap_box(wrap, ctx):
    """Where a battery can be, given its coloured wrapper: the wrapper's minimum-area
    rectangle, a little wider, and longer at both ends by the end caps. Grey shadow beside
    the battery lies outside this box and is not counted as battery."""
    pts = np.column_stack(np.nonzero(wrap))[:, ::-1].astype(np.float32)
    (cx, cy), (w, h), a = cv2.minAreaRect(pts)
    side, end = 0.1 * ctx.d_min, CAP * ctx.d_min
    if w >= h:
        w, h = w + 2 * end, h + 2 * side
    else:
        w, h = w + 2 * side, h + 2 * end
    box = np.zeros(wrap.shape, np.uint8)
    cv2.fillPoly(box, [np.round(cv2.boxPoints(((cx, cy), (w, h), a))).astype(np.int32)], 1)
    return box > 0


def colour_splits(mask, ctx):
    """Seeds from colour: the coloured wrapper of every battery (shadows between batteries are
    grey, so wrappers lying side by side stay apart) and the inner part of every grey or
    silver piece left over (a coin cell). The seeds are grown back to fill the outline."""
    if ctx.norm is None:
        return []
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    nm = ctx.norm[y0:y1, x0:x1]
    mn = nm.mean(axis=-1)
    chroma = np.sqrt(((nm - mn[..., None]) ** 2).sum(axis=-1)) / np.maximum(mn, 1e-3)
    m = mask[y0:y1, x0:x1]
    col0 = ((chroma > measure.COLOURED) & m).astype(np.uint8)
    col0 = cv2.morphologyEx(col0, cv2.MORPH_OPEN, _disc(2))
    if col0.sum() < ctx.min_piece:
        return []
    # what is left once the wrappers (and a band around them) are taken away: shadows are
    # thin and vanish in the erosion, a coin cell keeps a solid core
    rest = m & (cv2.dilate(col0, _disc(0.3 * ctx.d_min)) == 0)
    rest = cv2.erode(rest.astype(np.uint8), _disc(0.25 * ctx.d_min))
    n, rlab, rst, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    cores = [rlab == i for i in range(1, n) if rst[i, cv2.CC_STAT_AREA] >= 0.25 * ctx.min_piece]
    # two wrappers can touch (a battery's end pressed on another's side); shrinking the
    # coloured mask a little at a time parts them at that thin contact
    found, seen = [], set()
    r = 0.0
    while r <= SHRINK_MAX * ctx.d_min:
        col = cv2.erode(col0, _disc(r)) if r >= 1 else col0
        n, lab, st, _ = cv2.connectedComponentsWithStats(col, connectivity=8)
        wraps = [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.5 * ctx.min_piece]
        k = len(wraps) + len(cores)
        if k >= 2 and k not in seen:
            seen.add(k)
            seeds = np.zeros(mask.shape, np.int32)
            for j, s in enumerate(wraps + cores, start=1):
                seeds[y0:y1, x0:x1][s] = j
            pcs = _grow(seeds, mask)
            # a wrapper grows back only as far as its silver end caps reach; grey shadow
            # further out (the wedge between two batteries) is not part of the battery
            # first every shrunk seed takes back its own coloured wrapper (nearest seed), then
            # it may reach a little past the colour, as far as the end caps go
            col_full = np.zeros(mask.shape, bool)
            col_full[y0:y1, x0:x1] = col0 > 0
            wrap_seeds = np.where(seeds <= len(wraps), seeds, 0)
            full_wraps = _grow(wrap_seeds, col_full)
            for j in range(min(len(wraps), len(pcs), len(full_wraps))):
                pcs[j] = pcs[j] & _cap_box(full_wraps[j], ctx)
            if all(pc.sum() >= ctx.min_piece for pc in pcs):
                found.append(("colour", pcs))
        r += max(1.0, 0.05 * ctx.d_min)
    return found


def corner_splits(mask, ctx):
    """Cut between pairs of deep inward corners of the outline."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    if len(c) < 10:
        return []
    hull = cv2.convexHull(c, returnPoints=False)
    try:
        dfx = cv2.convexityDefects(c, hull)
    except cv2.error:
        return []
    if dfx is None:
        return []
    deep = [(d / 256.0, (int(c[f][0][0]), int(c[f][0][1]))) for _, _, f, d in dfx.reshape(-1, 4)
            if d / 256.0 >= DEFECT_MIN * ctx.d_min]
    deep = [p for _, p in sorted(deep, reverse=True)[:6]]
    found = []
    for a, b in itertools.combinations(deep, 2):
        L = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        if L < 3 or L > 1.3 * ctx.d_max:      # a cut is never wider than a battery
            continue
        line = np.zeros(mask.shape, np.uint8)
        cv2.line(line, a, b, 1, 3)
        if mask[line > 0].mean() < 0.85:        # the cut must run through the object
            continue
        cut = mask & (line == 0)
        n, lab = cv2.connectedComponents(cut.astype(np.uint8), connectivity=4)
        pcs = [lab == i for i in range(1, n) if (lab == i).sum() >= ctx.min_piece]
        if len(pcs) >= 2:
            # give the cut line back to the piece beside it, so no width is lost
            seeds = np.zeros(mask.shape, np.int32)
            for k, pc in enumerate(pcs, start=1):
                seeds[pc] = k
            _, near = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
            lut = np.zeros(near.max() + 1, np.int32)
            lut[near[seeds > 0]] = seeds[seeds > 0]
            found.append(("corner cut", _pieces(np.where(mask, lut[near], 0), mask)))
    return found


def _solve(mask, ctx, depth):
    """Pieces that are each one named battery, or None. Returns (pieces, how, score)."""
    if ctx.calls >= BUDGET or mask.sum() < ctx.min_piece:
        return None
    p = ctx.measure(mask)
    if _named(p):
        return [p], [], p["match"]["confidence"]
    if depth == 0:
        return None
    return _best_split(mask, ctx, depth)


def _best_split(mask, ctx, depth):
    """Try every candidate split; each piece must be solved with one level less."""
    best = None
    for how, pcs in colour_splits(mask, ctx) + shrink_splits(mask, ctx) + corner_splits(mask, ctx):
        out, hows, scores = [], [how], []
        for pc in pcs:
            r = _solve(pc, ctx, depth - 1)
            if r is None:
                break
            out += r[0]
            hows += r[1]
            scores.append(r[2])
        else:
            # the weakest piece decides; fewer pieces win a tie
            score = min(scores)
            if best is None or (score, -len(out)) > (best[2], -len(best[0])):
                best = (out, hows, score)
        if ctx.calls >= BUDGET:
            break
    return best


def _touching_groups(parts):
    """Group objects whose outlines touch (within 2 pixels)."""
    n = len(parts)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grown = [cv2.dilate(p["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for p in parts]
    for i in range(n):
        for j in range(i + 1, n):
            if (grown[i] & parts[j]["mask"]).any():
                parent[root(j)] = root(i)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(parts[i])
    return list(groups.values())


def resplit(parts, ref, gray, norm, catalogue):
    """Replace every unnamed object that can be taken apart into named batteries by those
    batteries. Returns (new parts list, notes)."""
    if ref is None or not parts:
        return parts, []
    out = [p for p in parts if p["match"]["status"] == "identified"]
    unknown = [p for p in parts if p["match"]["status"] != "identified"]
    notes = []
    next_label = max(p["label"] for p in parts) + 1
    # unnamed pieces that touch each other are joined first: the first split may have cut
    # through the wrong place (a coin cell cut in half, one half joined to a battery)
    groups = _touching_groups(unknown)
    jobs = []
    for g in groups:
        if len(g) > 1:
            jobs.append((g, True))
        else:
            jobs.append((g, False))
    for g, joined in jobs:
        union = np.zeros(g[0]["mask"].shape, bool)
        for p in g:
            union |= p["mask"]
        ctx = _Ctx(ref, gray, norm, catalogue)
        # a single unnamed object was measured already and is not one battery, so it goes
        # straight to the splits; a joined group may also turn out to be one battery
        best = _solve(union, ctx, MAX_DEPTH) if joined else _best_split(union, ctx, MAX_DEPTH)
        if best is None or (not joined and len(best[0]) < 2):
            if joined:
                # the joined group did not work: try every piece on its own instead
                for p in g:
                    jobs.append(([p], False))
            else:
                out += g
            continue
        pieces, hows, _ = best
        if not hows:
            hows = ["joining pieces cut apart by mistake"]
        names = ", ".join(q["match"]["name"] for q in pieces)
        way = " and ".join(sorted(set(hows)))
        for q in pieces:
            q["label"] = next_label
            next_label += 1
            q["match"]["split"] = f"taken apart from touching batteries ({way})"
        what = f"{len(g)} unnamed pieces were joined and" if joined else "an unnamed object was"
        notes.append(f"{what} taken apart into {len(pieces)} batteries ({names}) by {way}")
        out += pieces
    return out, notes
