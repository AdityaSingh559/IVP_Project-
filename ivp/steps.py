"""The thirteen pipeline steps of the Battery Counter, shown one by one on the website.

The test script (tools/test_batteries.py) runs exactly these steps, so the site shows the
same numbers as the tests. Each step measures the photo, applies or skips itself, and says
why. The concept lists name only what the step really runs.
"""

import numpy as np
import cv2
from scipy import ndimage

from . import core, crop, health, identify, invoice, measure, resplit, segment

BLUE = (29, 95, 224)
GOOD = (29, 95, 224)
AMBIG = (217, 119, 6)
UNKNOWN = (200, 40, 40)
LEAK = (200, 30, 150)      # magenta: leaking battery
CHECKC = (230, 170, 0)     # amber: small deposit, check by eye


# ----------------------------------------------------------------------------- drawing helpers


def u8(x):
    return core.to_u8(x)


def gray3(g):
    g = u8(g)
    return np.dstack([g, g, g])


def overlay(rgb, mask, color=BLUE, alpha=0.55):
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return u8(out)


def palette(n):
    rng = np.random.default_rng(5)
    hues = (np.arange(n) * 0.618034) % 1.0
    cols = []
    for hh in hues:
        rgb = core.hsv_to_rgb(np.array([[[hh, 0.65, 0.95]]], np.float32))[0, 0]
        cols.append(tuple(int(v * 255) for v in rgb))
    rng.shuffle(cols)
    return cols


def label_image(labels):
    n = int(labels.max())
    cols = np.array([(255, 255, 255)] + palette(max(n, 1)), np.uint8)
    out = cols[labels]
    out[labels == 0] = (243, 247, 255)
    return out


def draw_outline(img, mask, color, thick=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, thick, lineType=cv2.LINE_AA)


def put_tag(img, text, x, y, color=BLUE, scale=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x0, y0 = int(x - tw / 2) - 5, int(y - th / 2) - 5
    x0 = max(0, min(x0, img.shape[1] - tw - 10))
    y0 = max(0, min(y0, img.shape[0] - th - 10))
    cv2.rectangle(img, (x0, y0), (x0 + tw + 10, y0 + th + 10), color, -1)
    cv2.putText(img, text, (x0 + 5, y0 + th + 4), font, scale, (255, 255, 255), 2, cv2.LINE_AA)


def number_parts(parts):
    """Reading order: rows from top to bottom, then left to right."""
    if not parts:
        return
    h = np.median([p["bbox"][3] - p["bbox"][1] for p in parts]) or 40
    order = sorted(parts, key=lambda p: (int(p["cy"] // (1.5 * h)), p["cx"]))
    for k, p in enumerate(order, start=1):
        p["no"] = k


def status_colour(p):
    st = (p.get("match") or {}).get("status")
    return GOOD if st == "identified" else AMBIG if st == "ambiguous" else UNKNOWN


def annotated(rgb, parts, ref=None):
    img = rgb.copy()
    if ref is not None and ref.get("mask") is not None:
        draw_outline(img, ref["mask"], (120, 120, 120), 2)
    for p in parts:
        draw_outline(img, p["mask"], status_colour(p), 3)
    for p in parts:
        tag = str(p.get("no", p["label"]))
        m = p.get("match") or {}
        if m.get("status") == "identified" and m.get("name"):
            short = m["name"].split()[0]          # "AA", "AAA", "CR2016", "LR44"
            tag += f" {m['count']}x{short}" if m.get("count", 1) > 1 else f" {short}"
        put_tag(img, tag, p["cx"], p["cy"], status_colour(p), 0.6)
    for p in parts:
        hs = (p.get("health") or {}).get("status")
        if hs in ("damaged", "check"):
            col = LEAK if hs == "damaged" else CHECKC
            x0, y0, x1, y1 = p["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = int(0.5 * np.hypot(x1 - x0, y1 - y0)) + 8
            cv2.circle(img, (int(cx), int(cy)), r, col, 3, cv2.LINE_AA)
            put_tag(img, "LEAK!" if hs == "damaged" else "CHECK", cx, max(12, cy - r), col, 0.55)
    return img


# ----------------------------------------------------------------------------- steps


def s_load(c):
    rgb = c["rgb"]
    g = core.rgb_to_gray(rgb)
    lap = core.laplacian(g)
    v = float(lap.var())
    c["blur_var"] = v
    sharp = v >= 15.0
    h, w = rgb.shape[:2]
    facts = [
        {"label": "Size", "value": f"{w} x {h} px", "note": f"a matrix of {h} rows, {w} columns, 3 colour channels"},
        {"label": "Sharpness", "value": f"{v:.0f}", "note": "variance of the Laplacian; below 15 the edges are too soft to measure well",
         "flag": not sharp},
    ]
    summary = ("The photo is sharp enough to measure." if sharp else
               "Warning: the photo looks blurred, so sizes may be less accurate. A sharper photo would help.")
    return {"status": "applied", "summary": summary, "facts": facts,
            "aux": [("Laplacian (edges and fine detail)", gray3(np.clip(np.abs(lap) * 4, 0, 255)))]}


def s_crop(c):
    rgb = c["rgb"]
    res = crop.autocrop(rgb)
    c["crop"] = {k: v for k, v in res.items() if k in ("status", "angle", "size", "reason")}
    f = res.get("find")
    aux = []
    if f is not None:
        aux = [("Whiteness V(1-S): paper is bright and unsaturated", gray3(f["white_small"])),
               ("Largest white region after Otsu, closing and opening", gray3(f["paper_small"] * 255)),
               ("Share of white per row (blue) and per column (orange); the paper is where it stays above the grey 50% line",
                crop.profile_plot(f["rows"], f["cols"]))]
    if res["status"] != "applied":
        # the paper may still not fill every corner (a cloth or table edge): keep its outline so
        # objects lying off the paper can be left out later
        if f is not None and f["area_frac"] > 0.5:
            c["paper_small"] = (f["paper_small"], f["k"])
        return {"status": "skipped", "after": None, "aux": aux,
                "summary": "Auto-crop not needed: " + res["reason"] + "."}
    c["crop_map"] = {"angle": res["angle"], "offset": res["offset"], "in_shape": rgb.shape[:2]}
    c["rgb"] = res["image"]
    W, H = res["size"]
    turn = (f"turned by {res['angle']:.1f} deg to make it straight, then " if res["angle"] else "")
    return {"status": "applied", "before": crop.draw_quad(rgb, res["quad"]), "after": res["image"], "aux": aux,
            "summary": (f"Found the white paper ({100 * f['area_frac']:.0f}% of the photo). The photo was {turn}cut to a "
                        f"straight rectangle of {W} x {H} px around the paper. Nothing is stretched: the photo is taken "
                        "from straight above, so a rigid crop keeps every size true."),
            "facts": [{"label": "Paper found", "value": f"{100 * f['area_frac']:.0f}%", "note": "of the photo, by Otsu on whiteness"},
                      {"label": "Turn", "value": f"{res['angle']:.1f} deg", "note": "from the minimum-area rectangle of the paper"},
                      {"label": "New size", "value": f"{W} x {H} px", "note": "rows and columns that are at least half white"}]}


def s_clean(c):
    rgb = c["rgb"]
    c["smooth"] = segment.denoise(rgb)

    def noise(img):
        g = core.rgb_to_gray(img)
        lap = core.laplacian(g)
        return float(1.4826 * np.median(np.abs(lap - np.median(lap))) / np.sqrt(20))

    n0, n1 = noise(rgb), noise(c["smooth"])
    h, w = rgb.shape[:2]
    cy, cx, s = h // 2, w // 2, min(h, w) // 6
    win = (slice(cy - s, cy + s), slice(cx - s, cx + s))
    zoom = lambda im: cv2.resize(u8(im[win]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    return {"status": "applied", "before": rgb, "after": u8(c["smooth"]),
            "summary": f"Noise reduced from {n0:.2f} to {n1:.2f} grey levels with a 3x3 median filter and a light Gaussian blur.",
            "aux": [("Centre, enlarged, before", zoom(rgb)), ("Centre, enlarged, after", zoom(c["smooth"]))]}


def s_light(c):
    sm = c["smooth"]
    bg = segment.fit_background(sm)
    c["bg"] = bg
    bb = bg.mean(axis=-1)
    var = float(np.percentile(bb, 98) / max(np.percentile(bb, 2), 1) - 1)
    norm = sm / bg
    shown = u8(np.clip(norm * 225, 0, 255))
    even = var < 0.10
    summary = (f"Lighting changes by {100*var:.0f}% across the background. Dividing by the fitted background removes it, "
               "so an object in a dark corner looks the same as one in the light." if not even else
               f"Lighting is already fairly even (it changes by {100*var:.0f}%), so dividing by the background changes little.")
    return {"status": "applied", "before": u8(sm), "after": shown, "summary": summary,
            "facts": [{"label": "Lighting change", "value": f"{100*var:.0f}%", "note": "brightest against darkest part of the empty background",
                       "flag": not even}],
            "aux": [("Fitted background (the lighting without the objects)", u8(bg)),
                    ("Strongly coloured objects left out of the fit (HSV saturation)",
                     gray3(segment.coloured_objects(sm) * 255))]}


def s_find(c):
    om = segment.object_mask(c["smooth"], c["bg"])
    c["om"] = om
    shadow_px = float((om["shadow"] & (om["dist_u8"] > om["thresh"])).mean() * 100)
    return {"status": "applied", "before": c["rgb"], "after": overlay(c["rgb"], om["raw"]),
            "summary": (f"Pixels further than {om['thresh']/255:.2f} from the background colour are marked as objects "
                        f"(Otsu chose {om['otsu']/255:.2f}, the noise level allowed {om['thresh']/255:.2f}; the lower is used). "
                        f"{shadow_px:.1f}% of the photo was dark enough to look like an object but was recognised as shadow."),
            "facts": [{"label": "Otsu threshold", "value": f"{om['otsu']/255:.2f}", "note": "colour distance, from the histogram"},
                      {"label": "Threshold used", "value": f"{om['thresh']/255:.2f}", "note": "lower of Otsu and the noise-based bound"},
                      {"label": "Shadow removed", "value": f"{shadow_px:.1f}%", "note": "of the photo: only darker, same tint as the background"}],
            "aux": [("Colour distance from the background", gray3(om["dist_u8"].astype(np.float32) * 2)),
                    ("Shadow test (white = shadow)", gray3(om["shadow"] * 255))]}


def s_tidy(c):
    om = c["om"]
    filled = segment.fill_regions(om["raw"], om["dist"], om["thresh"] / 255.0)
    clean, notes = segment.detach(segment.tidy(filled))
    c["clean"] = clean
    comp, n = core.label_components(clean)
    c["components"] = comp
    raw_n = core.label_components(om["raw"])[1]
    holes = int(ndimage.label(ndimage.binary_fill_holes(clean) & ~clean)[1])
    return {"status": "applied", "before": gray3(om["raw"] * 255), "after": gray3(clean * 255),
            "summary": (f"Gaps inside objects were filled, openings that show the background were kept ({holes} openings), "
                        f"and specks were removed: {raw_n} raw pieces became {n} objects."
                        + (" Also: " + "; ".join(notes) + "." if notes else "")),
            "aux": [("Outlines by boundary extraction (A minus A eroded)", gray3(segment.boundary(clean) * 255))]}


def s_scale(c):
    comp = c["components"]
    ref = measure.find_reference(comp, c["rgb"])
    c["ref"] = ref
    c["parts_mask"] = c["clean"] & (comp != ref["label"]) if ref else c["clean"]
    img = c["rgb"].copy()
    if not ref:
        return {"status": "skipped", "after": None,
                "summary": ("No credit card was found, so sizes cannot be given in millimetres and nothing can be named. "
                            "Put the card flat on the paper, not touching anything, with its coloured side up.")}
    ref["mask"] = comp == ref["label"]
    draw_outline(img, ref["mask"], BLUE, 4)
    L, W = ref["px"]
    corners = np.array(ref["corners"])
    for p in corners:
        cv2.circle(img, tuple(int(v) for v in np.round(p)), 6, (220, 40, 40), -1, cv2.LINE_AA)
    dev = max(abs(a - 90) for a in ref["corner_angles"])
    facts = [{"label": "Card size", "value": f"{L:.1f} x {W:.1f} px", "note": "each side fitted as a straight line to sub-pixel edges"},
             {"label": "Aspect ratio", "value": f"{L / W:.3f}", "note": "a real card is exactly 1.586 (85.60 / 53.98 mm)"},
             {"label": "Corner angles", "value": " / ".join(f"{a:.1f}" for a in ref["corner_angles"]),
              "note": "90 deg on the real card; any difference is leftover perspective, removed by the homography", "flag": dev > 3},
             {"label": "Scale", "value": f"{ref['mm_per_px']:.4f} mm/px", "note": f"{1 / ref['mm_per_px']:.2f} px per mm (average)"}]
    summary = (f"Found the credit card: {L:.1f} x {W:.1f} px. A card is always 85.60 x 53.98 mm (ISO/IEC 7810 ID-1), "
               f"so one pixel is about {ref['mm_per_px']:.4f} mm. Its four corners (red) give a homography from pixels to "
               f"millimetres, so every battery outline is measured in true millimetres even if the photo was slightly tilted "
               f"(the card's corners were off 90 deg by up to {dev:.1f} deg).")
    return {"status": "applied", "before": c["rgb"], "after": img, "summary": summary, "facts": facts}


def s_split(c):
    labels, dt, peaks = segment.split_touching(c["parts_mask"])
    c["labels"] = labels
    comp, n = core.label_components(c["parts_mask"])
    split = 0
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        ids = np.unique(labels[sl][comp[sl] == i])
        if len(ids[ids > 0]) > 1:
            split += 1
    n_parts = len(np.unique(labels)) - 1
    dt_img = gray3(dt / max(dt.max(), 1) * 255)
    dt_img[cv2.dilate(peaks.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = (220, 40, 40)
    aux = [("Distance transform with markers (red)", dt_img)]
    if split == 0:
        return {"status": "skipped", "after": None, "aux": aux,
                "summary": f"No objects were touching: each of the {n_parts} objects already has one marker, so nothing was split."}
    return {"status": "applied", "before": gray3(c["parts_mask"] * 255), "after": label_image(labels), "aux": aux,
            "summary": f"{split} groups of touching objects were split by watershed, giving {n_parts} separate objects."}


def s_measure(c):
    parts = measure.measure_parts(c["labels"], c["ref"], c["smooth"], norm=c["smooth"] / c["bg"])
    # an object touching the photo's edge is not fully in view (often the paper's own edge
    # or a sheet behind it), so it cannot be measured and is left out
    H, W = c["rgb"].shape[:2]
    parts = [p for p in parts if p["bbox"][0] > 2 and p["bbox"][1] > 2 and p["bbox"][2] < W - 3 and p["bbox"][3] < H - 3]
    # with the scale known, anything narrower than half the smallest battery is not a battery
    # (a fold in the paper, a pencil mark, a speck) and is left out
    junk = 0
    # objects lying off the paper (a cloth or table corner the crop could not cut away)
    if "paper_small" in c:
        pm, kk = c["paper_small"]
        on = lambda p: pm[min(pm.shape[0] - 1, int(p["cy"] * kk)), min(pm.shape[1] - 1, int(p["cx"] * kk))]
        keep = [p for p in parts if on(p)]
        junk += len(parts) - len(keep)
        parts = keep
    if c["ref"]:
        smallest = min(r["second_mm"] or r["size_mm"] for r in c["catalogue"])
        width = lambda p: min(p["feret_min_mm"], p.get("cyl_d_mm") or p["feret_min_mm"])
        keep = [p for p in parts if width(p) >= 0.5 * smallest]
        junk += len(parts) - len(keep)
        parts = keep
    elif parts:
        # no card, so no millimetres: compare with the widest object instead. The narrowest
        # battery (LR44, 11.6 mm) is still 58% as wide as the widest (CR2016, 20 mm), so
        # anything under 35% of the widest is a paper edge, a fold or a speck
        widest = max(p["feret_min_px"] for p in parts)
        keep = [p for p in parts if p["feret_min_px"] >= 0.35 * widest]
        junk += len(parts) - len(keep)
        parts = keep
    for p in parts:
        p["type"] = identify.part_type(p)
    c["parts"] = parts
    img = c["rgb"].copy()
    for p in parts:
        draw_outline(img, p["mask"], BLUE, 2)
        if c["ref"]:
            tag = (f"D {p['eq_d_mm']:.1f}" if p["type"] == "button" else
                   f"{p['cyl_len_mm']:.1f} x {p['cyl_d_mm']:.1f}")
            put_tag(img, tag, p["cx"], p["cy"], BLUE, 0.45)
    details = []
    names = {"cylinder": "cylinder", "button": "round cell", "other": "other shape"}
    for p in parts:
        if not c["ref"]:
            break
        if p["type"] == "button":
            details.append(f"round cell at ({p['cx']:.0f}, {p['cy']:.0f}): diameter {p['eq_d_mm']:.2f} mm "
                           f"(circularity {p['circularity']:.2f})")
        else:
            details.append(f"{names[p['type']]} at ({p['cx']:.0f}, {p['cy']:.0f}): length {p['cyl_len_mm']:.2f} mm, "
                           f"diameter {p['cyl_d_mm']:.2f} mm (elongation {p['elongation']:.1f})")
    nc = sum(p["type"] == "cylinder" for p in parts)
    nb = sum(p["type"] == "button" for p in parts)
    summary = (f"Measured {len(parts)} objects in millimetres: {nc} cylinders (length x diameter) and {nb} round cells "
               "(diameter), written on each."
               + (f" {junk} thin marks, specks or objects off the paper were left out." if junk else "")
               if c["ref"] else
               f"Measured {len(parts)} objects in pixels only, because no card gave the scale."
               + (f" {junk} thin marks, specks or objects off the paper were left out." if junk else ""))
    return {"status": "applied", "before": c["rgb"], "after": img, "summary": summary, "details": details}


def s_identify(c):
    parts = c["parts"]
    identify.identify(parts, c["catalogue"], c["ref"] is not None)
    # second try for touching batteries: only objects that could not be named are touched
    parts, split_notes = resplit.resplit(parts, c["ref"], c["smooth"], c["smooth"] / c["bg"], c["catalogue"])
    c["parts"] = parts
    number_parts(parts)
    img = annotated(c["rgb"], parts, c["ref"])
    c["annotated"] = img
    st = {"identified": 0, "ambiguous": 0, "unknown": 0}
    for p in parts:
        st[p["match"]["status"]] += 1
    details = []
    for p in sorted(parts, key=lambda q: q["no"]):
        m = p["match"]
        if m["status"] == "identified" and m.get("count", 1) > 1:
            details.append(f"{p['no']}: {m['count']} x {m['name']} ({m['group']}, could not be split, counted by size; "
                           f"confidence {100 * m['confidence']:.0f}%)")
        elif m["status"] == "identified":
            details.append(f"{p['no']}: {m['name']} (confidence {100 * m['confidence']:.0f}%, "
                           f"distance {m['distance']:.2f} of 1.00 allowed"
                           + (f"; {m['split']}" if m.get("split") else "") + ")")
        elif m["status"] == "ambiguous":
            details.append(f"{p['no']}: ambiguous between {m['between'][0]} and {m['between'][1]} ({m['reason']})")
        else:
            details.append(f"{p['no']}: unknown ({m['reason']})")
    if not parts:
        return {"status": "skipped", "after": None, "summary": "There are no objects to identify."}
    return {"status": "applied", "before": c["rgb"], "after": img, "details": details,
            "summary": (f"{st['identified']} objects named from the catalogue, {st['ambiguous']} ambiguous, {st['unknown']} unknown. "
                        + ("Touching batteries: " + "; ".join(split_notes) + ". " if split_notes else "")
                        + "Blue outline: named. Orange: ambiguous. Red: unknown.")}


def s_health(c):
    parts = c["parts"]
    if not parts:
        return {"status": "skipped", "after": None, "summary": "There are no batteries to check."}
    k = c["ref"]["mm_per_px"] if c["ref"] else None
    card = c["ref"].get("mask") if c["ref"] else None
    res = health.check(c["rgb"], c["smooth"] / c["bg"], parts, k, card, c["bg"])
    c["health"] = {"counts": res["counts"], "tex_thr": res["tex_thr"]}
    img = annotated(c["rgb"], parts, c["ref"])
    c["annotated"] = img
    leak_img = overlay(c["rgb"], res["leak"], LEAK, 0.75)
    for p in parts:
        hs = p["health"]["status"]
        if hs != "ok":
            draw_outline(leak_img, p["mask"], LEAK if hs == "damaged" else CHECKC, 2)
    n = res["counts"]
    unit = "mm^2" if k else "% of the battery"
    lim = (f"{health.CHECK_MM2} / {health.DAMAGED_MM2} mm^2" if k else
           f"{100 * health.CHECK_FRAC:.1f} / {100 * health.DAMAGED_FRAC:.0f}% of the battery")
    details = []
    for p in sorted(parts, key=lambda q: q.get("no", 0)):
        h_ = p["health"]
        name = (p["match"].get("name") if p["match"]["status"] == "identified" else None) or "object"
        word = {"ok": "OK", "check": "CHECK", "damaged": "LEAKING"}[h_["status"]]
        amt = (f"{h_['area_mm2']:.1f} mm^2" if h_["area_mm2"] is not None else f"{100 * h_['frac']:.1f}%")
        details.append(f"{p.get('no', '?')}: {name} - {word} ({h_['reason'] if h_['status'] != 'ok' else 'deposit ' + amt})")
    if n["damaged"]:
        summary = (f"WARNING: {n['damaged']} {'battery looks' if n['damaged'] == 1 else 'batteries look'} leaking (magenta circle). "
                   "Do not use them; dispose of them at a battery recycling point. ")
    elif n["check"]:
        summary = f"{n['check']} {'battery shows' if n['check'] == 1 else 'batteries show'} a small deposit (amber circle): check by eye. "
    else:
        summary = "No leakage seen on any battery. "
    summary += (f"Checked {len(parts)} objects at their ends (cylinders) or face and rim (round cells): "
                f"{n['ok']} OK, {n['check']} to check, {n['damaged']} leaking.")
    zones = overlay(c["rgb"], res["zone"], (40, 160, 220), 0.35)
    tex = res["maps"]["tex"]
    tex_img = gray3(np.clip(tex / max(3 * res["tex_thr"], 1e-6), 0, 1) * 255)
    glare_img = gray3(np.clip(res["maps"]["tophat"] * 4, 0, 255))
    glare_img[res["maps"]["glare"]] = (230, 60, 60)
    return {"status": "applied", "before": c["rgb"], "after": leak_img, "summary": summary, "details": details,
            "facts": [{"label": "Leaking", "value": str(n["damaged"]), "note": f"deposit of {health.DAMAGED_MM2 if k else 100 * health.DAMAGED_FRAC} {unit} or more",
                       "flag": n["damaged"] > 0},
                      {"label": "To check", "value": str(n["check"]), "note": f"small deposit, between the two limits ({lim})",
                       "flag": n["check"] > 0},
                      {"label": "OK", "value": str(n["ok"]), "note": "no crust or corrosion seen"},
                      {"label": "Rough above", "value": f"{res['tex_thr']:.3f}",
                       "note": "texture threshold, set from this photo's own empty paper"}],
            "aux": [("Where a leak would show: cylinder ends, cell face and rim, and a ring of paper around them", zones),
                    ("Texture: local RMS of (image - 5x5 median); grain is bright, smooth metal and edges stay dark", tex_img),
                    ("Grey-scale top-hat; red = glare (brighter than the paper), never counted as crust", glare_img)]}


def s_invoice(c):
    inv = invoice.build(c["parts"], c["catalogue"])
    c["invoice"] = inv
    lines = len(inv["lines"])
    return {"status": "applied" if lines else "skipped", "after": None,
            "summary": (f"Counted {sum(l['quantity'] for l in inv['lines'])} batteries of {lines} kinds, grand total {inv['grand_total']}. "
                        f"{len(inv['not_charged'])} objects were not counted because they could not be named."
                        if lines else "Nothing could be named, so there is nothing to count."),
            "facts": [{"label": l["name"], "value": f"{l['quantity']} x {l['unit_price']}", "note": f"= {l['line_total']}"}
                      for l in inv["lines"]] + ([{"label": "Grand total", "value": inv["grand_total"], "note": "quantity x unit price, added"}]
                                                if lines else [])}


def s_export(c):
    if "annotated" not in c:
        c["annotated"] = annotated(c["rgb"], c["parts"], c["ref"])
    return {"status": "applied", "after": None,
            "summary": "The numbered photo and the battery count are ready to download as JPG, PDF and CSV on the Results page."}


STEPS = [
    {"id": "load", "title": "Load and check", "fn": s_load,
     "desc": "Reads the photo as a grid of numbers and checks that it is sharp enough to measure.",
     "syllabus": ["Image as a matrix", "Image file formats", "Spatial resolution", "Laplacian filter"], "extra": []},
    {"id": "crop", "title": "Auto-crop the paper", "fn": s_crop,
     "desc": "Finds the white paper and cuts a straight rectangle around it, so the table, laptop or cloth around it is ignored. The photo is only turned and cropped, never stretched.",
     "syllabus": ["RGB to HSV", "Otsu threshold", "Closing", "Opening", "Connected-component labelling",
                  "Projection profiles", "Rotation (geometric transform)"],
     "extra": ["Minimum-area rectangle"]},
    {"id": "clean", "title": "Clean up noise", "fn": s_clean,
     "desc": "Removes camera and compression noise before anything is measured.",
     "syllabus": ["Median filter", "Gaussian filter"], "extra": []},
    {"id": "light", "title": "Even out lighting", "fn": s_light,
     "desc": "Estimates the lighting of the empty paper and divides it out, so uneven light and dark corners no longer matter.",
     "syllabus": ["Gaussian filter", "RGB to HSV (saturation)", "Colour slicing", "Convex hull"],
     "extra": ["Background division (normalised convolution)"]},
    {"id": "find", "title": "Find the objects", "fn": s_find,
     "desc": "Marks every pixel whose colour differs from the background, and tells soft shadows apart from objects.",
     "syllabus": ["Euclidean distance", "Otsu threshold", "Thresholding", "Median filter"],
     "extra": ["Shadow detection (same tint, only darker)"]},
    {"id": "tidy", "title": "Tidy the mask", "fn": s_tidy,
     "desc": "Fills gaps inside objects, keeps openings that show the background, and removes specks.",
     "syllabus": ["Closing", "Opening", "Erosion", "Dilation", "Boundary extraction", "Connected-component labelling"],
     "extra": []},
    {"id": "scale", "title": "Find the scale", "fn": s_scale,
     "desc": "Finds the credit card (always 85.60 x 53.98 mm) by its rectangular shape and works out how many millimetres one pixel is.",
     "syllabus": ["Connected components", "Image gradient", "RGB to HSV (saturation)"],
     "extra": ["Minimum-area rectangle", "Sub-pixel edge localisation", "Robust line fitting (Huber)",
               "Metric rectification (card homography)"]},
    {"id": "split", "title": "Split touching objects", "fn": s_split,
     "desc": "Separates objects that touch, so each one is counted once.",
     "syllabus": ["Connected-component labelling (4 and 8)"],
     "extra": ["Distance transform", "h-maxima markers (morphological reconstruction)", "Watershed"]},
    {"id": "measure", "title": "Measure", "fn": s_measure,
     "desc": "Sorts each object into cylinder (AA, AAA) or round cell (CR2016, LR44) by its shape, and measures it in millimetres.",
     "syllabus": ["Region moments"],
     "extra": ["Feret diameters (rotating calipers)", "Shape descriptors: elongation, circularity",
               "Equivalent diameter", "Sub-pixel edge localisation"]},
    {"id": "identify", "title": "Identify", "fn": s_identify,
     "desc": ("Names each battery by comparing its size with the catalogue, and says how confident it is. "
              "An object that cannot be named is tried again as touching batteries: it is split by colour, by "
              "shrinking and growing back, or at its inward corners, and the split is kept only if every piece is named."),
     "syllabus": ["Erosion", "Connected-component labelling"],
     "extra": ["Nearest-size matching with a tolerance and an ambiguity margin",
               "Gaussian-likelihood confidence", "Hu moments (stored, not used to decide)",
               "Convexity defects", "Nearest-seed region growing (distance transform with labels)"]},
    {"id": "health", "title": "Leakage check", "fn": s_health,
     "desc": ("Looks for leaks on every battery: white crust (alkaline leakage), green-blue or rust-brown "
              "corrosion at the ends of a cylinder or on the face and rim of a round cell. A leaking battery "
              "is circled in magenta with a warning."),
     "syllabus": ["RGB to HSV", "Colour slicing", "Median filter", "Grey-scale top-hat", "Closing", "Opening",
                  "Morphological gradient", "Connected-component labelling"],
     "extra": ["Texture by local RMS of a median residual", "Glare test (brighter than paper)",
               "Body-colour exclusion", "Leak area in mm^2 from the card scale"]},
    {"id": "invoice", "title": "Count and total", "fn": s_invoice,
     "desc": "Counts each kind of battery, multiplies by its price and adds the total.",
     "syllabus": [], "extra": []},
    {"id": "export", "title": "Export", "fn": s_export,
     "desc": "Prepares the numbered photo and the battery count for download.",
     "syllabus": ["Image file formats (JPEG uses the DCT)"], "extra": ["PDF and CSV export"]},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

NOT_USED = ["Unit 8 (video)", "Hit-or-Miss transform", "Hadamard transform", "Bayesian classification", "EM",
            "Hough circle transform (the card gives the scale, not a coin)",
            "Region splitting and merging", "Contrast stretching, histogram equalization and CLAHE (not needed after background division)",
            "Homomorphic filtering (background division is used instead)", "Canny edge detector", "Hough lines"]
