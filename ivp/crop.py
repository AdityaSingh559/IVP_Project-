"""Auto-crop: find the white paper in the photo and cut out a straight rectangle around it.

The photo is taken from straight above (90 degrees), so the paper is a rectangle in the photo,
only possibly turned a little. The crop is therefore rigid: rotate, then cut a rectangle. No
perspective warp is used, so nothing in the photo is ever stretched.

  1. whiteness  W = V (1 - S) from HSV: paper is bright AND unsaturated; a wooden table,
                a laptop, cloth or skin are darker or coloured
  2. threshold  Otsu on W, but never below 72% of the brightest white (so a grey laptop or
                desk is not taken for paper), then an opening that cuts thin links
  3. angle      the minimum-area rectangle of the largest white region gives the paper's turn
  4. edges      the mask is turned straight; then, for every row and every column, the share
                of white pixels is counted (projection profiles). The paper is the longest run
                of rows and of columns that are at least half white. Curled corners, a bent
                page edge or a white blob next to the paper hardly change these counts, so the
                result is always a clean rectangle
  5. crop       the full-size photo is turned by the same angle and the rectangle is cut out
If the paper already fills the photo, nothing is cropped.
"""
import cv2
import numpy as np

from . import core

WORK = 700           # long side of the copy used to find the paper
COVER = 0.5          # a row / column belongs to the paper when at least this share is white
PAPER_REL = 0.62     # paper is at least this bright relative to the photo's brightest white


def whiteness(rgb):
    f = rgb.astype(np.float32) / 255.0
    mx, mn = f.max(-1), f.min(-1)
    s = (mx - mn) / np.maximum(mx, 1e-3)
    return mx * (1 - s)


def longest_run(on):
    """Start and end (exclusive) of the longest run of True values."""
    best, start, b0, b1 = 0, None, 0, 0
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, b0, b1 = i - start, start, i
            start = None
    return b0, b1


def rotate(img, angle_deg, interp=cv2.INTER_LINEAR):
    """Turn an image about its centre, growing the canvas so nothing is cut off.
    Returns the turned image and the 2x3 matrix used."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    M[0, 2] += W / 2 - w / 2
    M[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE), M


def find_paper(rgb):
    h, w = rgb.shape[:2]
    k = WORK / max(h, w)
    small = cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    W = cv2.GaussianBlur(whiteness(small), (0, 0), 1.5)
    wu8 = core.to_u8(W * 255)
    # Otsu splits dark from light; a grey laptop or desk is "light" next to a black card, so
    # paper must also be close to the brightest white in the photo (its 90th percentile)
    t = max(core.otsu_threshold(wu8), int(PAPER_REL * np.percentile(wu8, 90)))
    paper = (wu8 > t).astype(np.uint8)
    # closing fills the objects lying on the paper; opening cuts thin links to other white things
    s = max(5, int(0.03 * WORK)) | 1
    m = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    # region filling: everything lying on the paper (the card, the batteries) becomes paper,
    # so a large dark card does not make its rows and columns look empty
    # a card lying close to the paper's edge leaves only a thin strip of paper beside it; a
    # wider closing seals that strip first, so the card counts as enclosed
    seal = int(0.06 * WORK) | 1
    sealed = cv2.morphologyEx((lab == i).astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal, seal)))
    cs, _ = cv2.findContours(sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(cs, key=cv2.contourArea)
    comp = np.zeros(m.shape, np.uint8)
    cv2.drawContours(comp, [outline], -1, 1, -1)
    # the paper's turn, from its minimum-area rectangle, folded into -45..45 degrees
    (_, _), (_, _), a = cv2.minAreaRect(outline)
    angle = float(((a + 45) % 90) - 45)
    if abs(angle) < 0.5:
        angle = 0.0
    # projection profiles of the straightened mask
    straight, M = rotate(comp * 255, angle, cv2.INTER_NEAREST)
    straight = straight > 127
    rows, cols = straight.mean(axis=1), straight.mean(axis=0)
    y0, y1 = longest_run(rows >= COVER * rows.max())
    x0, x1 = longest_run(cols >= COVER * cols.max())
    # corners of that rectangle, taken back into the photo's own coordinates (for drawing)
    Minv = cv2.invertAffineTransform(M)
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    quad = (box @ Minv[:, :2].T + Minv[:, 2]) / k
    area = (x1 - x0) * (y1 - y0) / (small.shape[0] * small.shape[1])
    return {"angle": angle, "box_small": (x0, y0, x1, y1), "k": k, "quad": quad, "area_frac": area,
            "paper_small": comp.astype(bool), "white_small": wu8, "otsu": t,
            "rows": rows, "cols": cols}


def autocrop(rgb):
    """Returns dict(status, image, quad, ...). status 'applied' or 'skipped'."""
    h, w = rgb.shape[:2]
    f = find_paper(rgb)
    if f is None or f["area_frac"] < 0.15:
        return {"status": "skipped", "image": rgb, "find": f,
                "reason": "no sheet of white paper was found, so the whole photo is used"}
    x0, y0, x1, y1 = f["box_small"]
    sh, sw = f["paper_small"].shape
    full = x0 <= 1 and y0 <= 1 and x1 >= sw - 1 and y1 >= sh - 1
    if f["area_frac"] > 0.96 or (full and f["angle"] == 0.0):
        return {"status": "skipped", "image": rgb, "find": f,
                "reason": f"the paper already fills the photo ({100 * f['area_frac']:.0f}%), so nothing needs cropping"}
    # same turn on the full-size photo, then cut the rectangle (scaled up from the small copy)
    big, _ = rotate(rgb, f["angle"]) if f["angle"] else (rgb, None)
    k = f["k"]
    # the small copy was turned on its own canvas; map the box through the canvas centres
    sh_r, sw_r = rotate(np.zeros((sh, sw), np.uint8), f["angle"])[0].shape if f["angle"] else (sh, sw)
    ox, oy = (big.shape[1] - sw_r / k) / 2, (big.shape[0] - sh_r / k) / 2
    X0, X1 = x0 / k + ox, x1 / k + ox
    Y0, Y1 = y0 / k + oy, y1 / k + oy
    # trim a thin strip so the paper's own edge and its shadow are not seen as an object
    m = max(4, int(0.008 * max(X1 - X0, Y1 - Y0)))
    X0, Y0 = int(np.clip(X0 + m, 0, big.shape[1])), int(np.clip(Y0 + m, 0, big.shape[0]))
    X1, Y1 = int(np.clip(X1 - m, 0, big.shape[1])), int(np.clip(Y1 - m, 0, big.shape[0]))
    out = big[Y0:Y1, X0:X1].copy()
    return {"status": "applied", "image": out, "quad": f["quad"], "angle": f["angle"],
            "size": (out.shape[1], out.shape[0]), "find": f, "kept_frac": out.size / rgb.size,
            "offset": (X0, Y0)}


def line_intersect(l1, l2):
    """Intersection of two lines given as (vx, vy, x0, y0), as cv2.fitLine returns them."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1]))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


def draw_quad(rgb, q, colour=(29, 95, 224)):
    img = rgb.copy()
    t = max(3, int(0.004 * max(img.shape[:2])))
    cv2.polylines(img, [np.round(q).astype(np.int32)], True, colour, t, cv2.LINE_AA)
    for p in q:
        cv2.circle(img, tuple(int(v) for v in np.round(p)), 3 * t, colour, -1, cv2.LINE_AA)
    return img


def profile_plot(rows, cols, w=720, h=160):
    """Row and column white-share profiles with the 50% line, for the website."""
    img = np.full((h, w, 3), 255, np.uint8)
    for k, (p, colour) in enumerate(((rows, (29, 95, 224)), (cols, (217, 119, 6)))):
        p = np.asarray(p, np.float32)
        xs = np.linspace(5, w - 5, len(p))
        ys = h - 10 - p / max(p.max(), 1e-6) * (h - 20)
        cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    y = int(h - 10 - COVER * (h - 20))
    cv2.line(img, (0, y), (w, y), (150, 150, 150), 1, cv2.LINE_AA)
    return img
