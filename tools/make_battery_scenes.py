"""Draw test photos of batteries on white paper with known contents (ground truth).

Each scene: an A4 sheet with a credit card and 4 AA, 4 AAA, 2 CR2016, 2 LR44 cells, put on a
wooden table and seen by a tilted camera (perspective), with uneven light, soft shadows,
blur, noise and JPEG compression. Writes data/tests/battery_*.jpg and battery_truth.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tests"
PX = 6.0                       # pixels per mm on the flat sheet before the camera warp
A4 = (210, 297)
SIZES = {"AA battery": (50.5, 14.5), "AAA battery": (44.5, 10.5),
         "CR2016 coin cell": (20.0, None), "LR44 button cell": (11.6, None)}
PRICES = {"AA battery": 20, "AAA battery": 20, "CR2016 coin cell": 35, "LR44 button cell": 15}


def wood(h, w, rng):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = np.sin(x / 9 + 3 * np.sin(y / 140) + rng.uniform(0, 6)) * 0.5 + 0.5
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    t = 0.65 * grain + 0.35 * (noise * 0.5 + 0.5)
    base = np.array([60, 105, 150], np.float32)       # BGR brown
    img = base[None, None] * (0.75 + 0.35 * t[..., None])
    return np.clip(img, 0, 255)


def shadow(canvas, mask, off=(6, 8), blur=9, depth=0.18):
    m = mask.astype(np.float32)
    M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
    s = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
    s = cv2.GaussianBlur(s, (0, 0), blur) * (1 - m)
    canvas *= (1 - depth * s)[..., None]


CRUST = {"white": (222, 236, 244), "green": (150, 172, 96)}   # BGR: alkaline crust, copper corrosion


def crust(canvas, cx, cy, area_mm2, kind, rng):
    """A leak deposit of about area_mm2 at (cx, cy): a cluster of small grains, rough and
    pitted, part on the battery and part spilled onto the paper. Returns its true area (mm^2)."""
    h, w = canvas.shape[:2]
    A = area_mm2 * PX * PX
    R = np.sqrt(A / np.pi)
    s = int(3 * R + 20)
    x0, y0 = max(0, int(cx) - s), max(0, int(cy) - s)
    x1, y1 = min(w, int(cx) + s), min(h, int(cy) + s)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for _ in range(600):
        if m.sum() >= A:
            break
        ox, oy = rng.normal(0, 0.55 * R, 2)
        r = max(1, int(round(rng.uniform(0.3, 0.8) * PX)))
        cv2.circle(m, (int(cx - x0 + ox), int(cy - y0 + oy)), r, 1, -1)
    win = canvas[y0:y1, x0:x1]
    yy, xx = np.nonzero(m)
    base = np.array(CRUST[kind], np.float32)
    grain = rng.normal(1.0, 0.09, len(yy))
    grain[rng.random(len(yy)) < 0.12] *= 0.7                      # pits
    win[yy, xx] = np.clip(base[None] * grain[:, None], 0, 255)
    return float(m.sum()) / (PX * PX)


def cylinder(canvas, cx, cy, ang, L, D, body_bgr, rng):
    """Battery lying on its side: shaded body, metal ends, + nub."""
    h, w = canvas.shape[:2]
    Lp, Dp = L * PX, D * PX
    nub_l, nub_d = 1.2 * PX, 0.38 * Dp
    body_l = Lp - nub_l
    mask = np.zeros((h, w), np.uint8)
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    def rect(a0, a1, half):
        pts = [c + u * a0 + v * half, c + u * a1 + v * half, c + u * a1 - v * half, c + u * a0 - v * half]
        return np.round(pts).astype(np.int32)
    a0 = -Lp / 2
    cv2.fillPoly(mask, [rect(a0, a0 + body_l, Dp / 2)], 1)
    cv2.fillPoly(mask, [rect(a0 + body_l - 1, Lp / 2, nub_d / 2)], 1)
    shadow(canvas, mask, blur=7, depth=0.25)
    yy, xx = np.nonzero(mask)
    P = np.column_stack([xx, yy]).astype(np.float32) - c
    along, across = P @ u, P @ v
    shade = 0.55 + 0.45 * np.cos(np.clip(across / (Dp / 2), -1, 1) * 1.35)       # round body
    spec = np.exp(-((across + 0.25 * Dp) / (0.08 * Dp)) ** 2) * 0.5              # highlight line
    col = np.tile(np.array(body_bgr, np.float32), (len(yy), 1))
    metal = (along > a0 + body_l - 0.1 * body_l) | (along < a0 + 0.04 * body_l)
    col[metal] = (175, 178, 182)
    val = col * shade[:, None] + 255 * spec[:, None]
    canvas[yy, xx] = np.clip(val, 0, 255)
    return mask


def disc(canvas, cx, cy, D, rng):
    """Coin or button cell lying flat: silver disc with a darker bevelled rim."""
    h, w = canvas.shape[:2]
    r = D * PX / 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1, cv2.LINE_AA)
    shadow(canvas, mask, off=(4, 5), blur=5, depth=0.22)
    yy, xx = np.nonzero(mask)
    d = np.hypot(xx - cx, yy - cy) / r
    base = 150 + 50 * np.cos((xx - cx + yy - cy) / r * 1.2)                        # brushed sheen
    rim = np.clip((d - 0.86) / 0.14, 0, 1)
    val = base * (1 - 0.45 * rim)
    canvas[yy, xx] = np.clip(np.column_stack([val + 6, val + 3, val]), 0, 255)
    # embossed text: a few darker strokes
    for k in range(3):
        y0 = int(cy - 0.25 * r + k * 0.2 * r)
        cv2.line(canvas, (int(cx - 0.4 * r), y0), (int(cx + 0.4 * r), y0), (120, 120, 118), max(1, int(0.04 * r)))
    return mask


def card(canvas, cx, cy, ang, rng):
    h, w = canvas.shape[:2]
    size = (85.60 * PX, 53.98 * PX)
    box = cv2.boxPoints(((cx, cy), size, np.degrees(ang)))
    mask = np.zeros((h, w), np.uint8)
    # rounded corners (3.18 mm radius)
    rr = int(3.18 * PX)
    tmp = np.zeros((int(size[1]) + 2, int(size[0]) + 2), np.uint8)
    cv2.rectangle(tmp, (rr, 0), (tmp.shape[1] - rr, tmp.shape[0]), 1, -1)
    cv2.rectangle(tmp, (0, rr), (tmp.shape[1], tmp.shape[0] - rr), 1, -1)
    for px_, py_ in ((rr, rr), (tmp.shape[1] - rr, rr), (rr, tmp.shape[0] - rr), (tmp.shape[1] - rr, tmp.shape[0] - rr)):
        cv2.circle(tmp, (px_, py_), rr, 1, -1)
    src = np.float32([[0, 0], [tmp.shape[1], 0], [tmp.shape[1], tmp.shape[0]]])
    dst = np.float32([box[1], box[2], box[3]])
    M = cv2.getAffineTransform(src, dst)
    mask = cv2.warpAffine(tmp, M, (w, h), flags=cv2.INTER_NEAREST)
    shadow(canvas, mask, blur=8, depth=0.2)
    yy, xx = np.nonzero(mask)
    t = (xx - xx.min()) / max(1, np.ptp(xx))
    col = np.column_stack([150 + 60 * t, 70 + 30 * t, 20 + 10 * t])                 # blue card (BGR)
    canvas[yy, xx] = col
    # chip
    chip = np.zeros_like(mask)
    ccx, ccy = cx - 0.28 * size[0] * np.cos(ang), cy - 0.28 * size[0] * np.sin(ang)
    cv2.fillPoly(chip, [cv2.boxPoints(((ccx, ccy), (11 * PX, 9 * PX), np.degrees(ang))).astype(np.int32)], 1)
    canvas[chip > 0] = (90, 180, 210)
    return mask


def scene(seed, touching=False, tilt=0.12, light=0.35, leaks=None, info=None):
    """leaks: list of {"name", "kind", "area"} - the first battery of that name gets a leak
    deposit at one end (cylinder) or on its rim (round cell). info (dict) receives the leak
    positions in photo pixels and their true areas."""
    rng = np.random.default_rng(seed)
    crng = np.random.default_rng(seed + 999)
    pending = list(leaks or [])
    placed_leaks = []
    W, H = int(A4[0] * PX), int(A4[1] * PX)
    sheet = np.full((H, W, 3), 238.0, np.float32)
    sheet += rng.normal(0, 1.5, sheet.shape)
    occ = np.zeros((H, W), np.uint8)
    truth = []

    def free(mask, margin=10):
        grown = cv2.dilate(mask, np.ones((margin, margin), np.uint8))
        return not (grown & occ).any() and not mask[:12].any() and not mask[-12:].any() \
            and not mask[:, :12].any() and not mask[:, -12:].any()

    def place(draw, tries=400, allow_touch=False):
        for _ in range(tries):
            cx, cy = rng.uniform(60, W - 60), rng.uniform(60, H - 60)
            ang = rng.uniform(0, np.pi)
            test = np.zeros_like(occ)
            m = draw(test.copy()[..., None].repeat(3, -1).astype(np.float32), cx, cy, ang)
            if free(m, 2 if allow_touch else 14):
                return draw(sheet, cx, cy, ang), (cx, cy, ang)
        raise RuntimeError("could not place")

    m, _ = place(lambda cv, x, y, a: card(cv, x, y, a, rng))
    occ |= m
    bodies = [(30, 30, 30), (40, 120, 200), (40, 140, 60), (150, 60, 40), (30, 30, 160)]
    items = ["AA battery"] * 4 + ["AAA battery"] * 4 + ["CR2016 coin cell"] * 2 + ["LR44 button cell"] * 2
    rng.shuffle(items)
    if touching:
        # two AA side by side, touching along their length, and an LR44 against an AAA
        L, D = SIZES["AA battery"]
        col = bodies[1]
        for _ in range(400):
            cx, cy, ang = rng.uniform(80, W - 80), rng.uniform(80, H - 80), rng.uniform(0, np.pi)
            v = np.array([-np.sin(ang), np.cos(ang)]) * D * PX
            t1 = cylinder(np.zeros((H, W, 3), np.float32), cx, cy, ang, L, D, col, rng)
            t2 = cylinder(np.zeros((H, W, 3), np.float32), cx + v[0], cy + v[1], ang, L, D, col, rng)
            if free(t1 | t2, 14):
                occ |= cylinder(sheet, cx, cy, ang, L, D, col, rng)
                occ |= cylinder(sheet, cx + v[0], cy + v[1], ang, L, D, (30, 30, 160), rng)
                truth += ["AA battery", "AA battery"]
                items.remove("AA battery")
                items.remove("AA battery")
                break
    for name in items:
        L, D = SIZES[name]
        if D:
            col = bodies[rng.integers(len(bodies))]
            m, (x, y, a) = place(lambda cv, x, y, a: cylinder(cv, x, y, a, L, D, col, rng))
        else:
            m, (x, y, a) = place(lambda cv, x, y, a: disc(cv, x, y, L, rng))
        occ |= m
        truth.append(name)
        spec = next((s for s in pending if s["name"] == name), None)
        if spec:
            pending.remove(spec)
            if D:   # the + end of the cylinder, a little inside the tip
                px_, py_ = x + np.cos(a) * (L / 2 - 1.5) * PX, y + np.sin(a) * (L / 2 - 1.5) * PX
            else:   # on the rim of the round cell
                px_, py_ = x + np.cos(a) * 0.42 * L * PX, y + np.sin(a) * 0.42 * L * PX
            true_area = crust(sheet, px_, py_, spec["area"], spec["kind"], crng)
            placed_leaks.append({**spec, "true_area": round(true_area, 2), "sheet_xy": (px_, py_)})
    # camera: sheet on a table, tilted view
    CW, CH = 1600, 1200
    table = wood(CH, CW, rng)
    s = 0.80 * CH / W          # the sheet fills most of the frame, as in a phone photo
    pw, ph = W * s, H * s
    x0, y0 = (CW - ph) / 2, (CH - pw) / 2           # sheet turned landscape on the table
    dst = np.float32([[x0 + ph, y0], [x0 + ph, y0 + pw], [x0, y0 + pw], [x0, y0]])
    j = tilt * pw
    dst += np.float32([[-j, -0.3 * j], [-0.2 * j, 0.4 * j], [0.3 * j, 0.2 * j], [0.6 * j, -0.5 * j]]) * rng.uniform(0.5, 1.0)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, Hm, (CW, CH), flags=cv2.INTER_LINEAR)
    pmask = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (CW, CH))
    shadow(table, (pmask > 0.5).astype(np.uint8), off=(10, 12), blur=12, depth=0.35)
    img = table * (1 - pmask[..., None]) + warped * pmask[..., None]
    # uneven light, blur, noise, JPEG
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    lx, ly = rng.uniform(0.2, 0.8) * CW, rng.uniform(0.2, 0.8) * CH
    g = 1 - light * (np.hypot(xx - lx, yy - ly) / np.hypot(CW, CH))
    img = img * g[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img += rng.normal(0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if info is not None:
        for lk in placed_leaks:
            q = cv2.perspectiveTransform(np.float32([[lk["sheet_xy"]]]), Hm)[0, 0]
            lk["photo_xy"] = (float(q[0]), float(q[1]))
        info["leaks"] = placed_leaks
    return img, truth


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [("battery_1_tilted.jpg", dict(seed=11)),
            ("battery_2_touching.jpg", dict(seed=22, touching=True)),
            ("battery_3_strong_tilt_dim.jpg", dict(seed=33, tilt=0.2, light=0.5))]
    truth = {}
    for name, cfg in cfgs:
        img, t = scene(**cfg)
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        counts = {k: t.count(k) for k in SIZES}
        truth[name] = {"counts": counts, "total": sum(PRICES[k] * v for k, v in counts.items())}
        print(name, counts, truth[name]["total"])
    (OUT / "battery_truth.json").write_text(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main()
