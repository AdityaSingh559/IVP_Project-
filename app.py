"""Battery Counter - FastAPI backend.

Serves the site in ./static and a JSON API that runs the thirteen pipeline steps of ivp/steps.py
on an uploaded photo, one request per step, so the page can tick each step off as it finishes."""

import copy
import csv
import datetime as dt
import io
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ivp import catalogue as catmod
from ivp.health import ADVICE as HEALTH_ADVICE
from ivp.steps import NOT_USED, STEP_BY_ID, STEPS

ROOT = Path(__file__).resolve().parent
MAX_SIDE = 1600
MAX_BYTES = 20 * 1024 * 1024
MAX_SESSIONS = 8

app = FastAPI(title="Battery Counter")

# ----------------------------------------------------------------------------- catalogue (shared)

CAT_LOCK = threading.Lock()
CATALOGUE = {"rows": [], "source": "", "errors": []}


SAMPLES = {"battery": ("battery_catalogue.csv", "battery_catalogue.csv (AA, AAA, CR2016, LR44)")}


def load_sample(name="battery"):
    fn, label = SAMPLES[name]
    rows, errs = catmod.load(ROOT / "data" / fn)
    CATALOGUE.update(rows=rows, source=label, errors=errs)


load_sample()


def cat_view():
    rows = [{"index": i, "name": r["name"], "type": r["type"], "size_mm": r["size_mm"],
             "second_mm": r["second_mm"], "price": f"{r['price']:.2f}"}
            for i, r in enumerate(CATALOGUE["rows"])]
    return {"rows": rows, "source": CATALOGUE["source"], "errors": CATALOGUE["errors"]}


@app.get("/api/catalogue")
def get_catalogue():
    return cat_view()


@app.post("/api/catalogue")
def upload_catalogue(file: UploadFile = File(...)):
    data = file.file.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(413, "The catalogue file is larger than 2 MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "The catalogue must be a UTF-8 CSV file.")
    rows, errs = catmod.parse_csv(text)
    if not rows:
        raise HTTPException(400, "No usable rows: " + ("; ".join(errs) if errs else "the file is empty."))
    with CAT_LOCK:
        CATALOGUE.update(rows=rows, source=file.filename or "uploaded.csv", errors=errs)
    return cat_view()


@app.post("/api/catalogue/sample")
def reset_catalogue(name: str = "battery"):
    if name not in SAMPLES:
        raise HTTPException(404, "No such sample catalogue.")
    with CAT_LOCK:
        load_sample(name)
    return cat_view()


def read_image(data):
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 20 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").upper()
        im.load()
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    if fmt not in {"JPEG", "PNG", "TIFF", "WEBP", "BMP", "MPO"}:
        raise HTTPException(415, f"Unsupported format {fmt or 'unknown'}. Use JPG, PNG, TIFF, WebP or BMP.")
    im = ImageOps.exif_transpose(im)
    ow, oh = im.size
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    im = im.convert("RGB")
    if min(im.size) < 200:
        raise HTTPException(400, "The image is too small (at least 200 px on the short side is needed).")
    if max(im.size) > MAX_SIDE:
        k = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8).copy()
    return arr, {"format": "JPEG" if fmt == "MPO" else fmt, "file_kb": round(len(data) / 1024, 1),
                 "orig_size": [ow, oh], "size": [arr.shape[1], arr.shape[0]],
                 "resized": (ow, oh) != (arr.shape[1], arr.shape[0])}


@app.get("/api/catalogue/sample.csv")
def sample_csv():
    return FileResponse(ROOT / "data" / "battery_catalogue.csv", media_type="text/csv", filename="battery_catalogue.csv")


# ----------------------------------------------------------------------------- sessions


class Session:
    def __init__(self, rgb, info):
        self.id = uuid.uuid4().hex[:12]
        self.info = info
        self.imgs = {}
        with CAT_LOCK:
            cat = copy.deepcopy(CATALOGUE["rows"])
        self.ctx = {"rgb": rgb, "catalogue": cat}
        self.original_key = self.put(rgb)
        self.idx = 0
        self.records = {}
        self.created = dt.datetime.now()

    def put(self, arr, png=False):
        key = uuid.uuid4().hex[:10]
        buf = io.BytesIO()
        im = Image.fromarray(np.ascontiguousarray(arr))
        im.save(buf, "PNG" if png else "JPEG", **({} if png else {"quality": 90}))
        self.imgs[key] = (buf.getvalue(), "image/png" if png else "image/jpeg")
        return key

    def url(self, key):
        return f"/api/s/{self.id}/img/{key}"


SESSIONS = OrderedDict()
LOCK = threading.Lock()


def get_session(sid):
    with LOCK:
        s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "This session has expired. Please upload the photo again.")
    return s


def step_meta(s):
    return {k: s[k] for k in ("id", "title", "desc", "syllabus", "extra")}


def concepts(ids=None):
    syl, ext = ["Image file formats", "Image as a matrix"], []
    for st in STEPS:
        if ids is None or st["id"] in ids:
            syl += [c for c in st["syllabus"] if c not in syl]
            ext += [c for c in st["extra"] if c not in ext]
    return syl, ext


@app.get("/api/steps")
def list_steps():
    syl, ext = concepts()
    return {"steps": [step_meta(s) for s in STEPS], "syllabus": syl, "extra": ext, "not_used": NOT_USED}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    arr, info = read_image(file.file.read(MAX_BYTES + 1))
    sess = Session(arr, info)
    info["reference"] = "credit card, 85.60 x 53.98 mm"
    with LOCK:
        SESSIONS[sess.id] = sess
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    return {"session": sess.id, "info": info, "original_url": sess.url(sess.original_key),
            "steps": [step_meta(s) for s in STEPS], "catalogue_rows": len(sess.ctx["catalogue"])}


@app.get("/api/s/{sid}/img/{key}")
def get_img(sid: str, key: str):
    s = get_session(sid)
    item = s.imgs.get(key)
    if item is None:
        raise HTTPException(404, "Image not found.")
    return Response(item[0], media_type=item[1], headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/s/{sid}/step/{step}")
def run_step(sid: str, step: str):
    s = get_session(sid)
    if step not in STEP_BY_ID:
        raise HTTPException(404, "Unknown step.")
    if s.idx >= len(STEPS) or STEPS[s.idx]["id"] != step:
        raise HTTPException(409, "That step is not the next one in the pipeline.")
    t0 = time.time()
    try:
        res = STEP_BY_ID[step]["fn"](s.ctx)
    except Exception as exc:
        s.idx = len(STEPS)  # the pipeline cannot continue after a failed step
        raise HTTPException(500, f"The step '{STEP_BY_ID[step]['title']}' failed ({type(exc).__name__}: {exc}).")
    out = {"step": step, "status": res["status"], "summary": res["summary"], "seconds": round(time.time() - t0, 2),
           "facts": res.get("facts", []), "details": res.get("details", [])}
    if res.get("after") is not None:
        before = res.get("before")
        out["before_url"] = s.url(s.put(before)) if before is not None else s.url(s.original_key)
        out["after_url"] = s.url(s.put(res["after"], png=res["after"].ndim == 3 and step in ("tidy", "split")))
    out["aux"] = [{"label": lbl, "url": s.url(s.put(im, png=True))} for lbl, im in res.get("aux", [])]
    s.records[step] = {"status": res["status"], "summary": res["summary"]}
    s.idx += 1
    nxt = STEPS[s.idx]["id"] if s.idx < len(STEPS) else None
    out.update({"next": nxt, "done": nxt is None})
    return out


def need_done(s):
    if s.idx < len(STEPS) or "invoice" not in s.ctx:
        raise HTTPException(409, "Run the pipeline first.")


TYPE_NAME = {"cylinder": "cylinder cell", "button": "round cell", "other": "other shape"}


def part_row(p, has_scale):
    m = p["match"]
    sizes = {}
    if has_scale:
        if p["type"] == "cylinder":
            sizes = {"length": p.get("cyl_len_mm"), "diameter": p.get("cyl_d_mm")}
        elif p["type"] == "button":
            sizes = {"diameter": p.get("eq_d_mm")}
        else:
            sizes = {"length": p.get("feret_max_mm"), "width": p.get("feret_min_mm")}
    hl = p.get("health") or {}
    return {"no": p["no"], "type": TYPE_NAME.get(p["type"], p["type"]),
            "health": hl.get("status", "ok"), "health_reason": hl.get("reason", ""),
            "leak_mm2": hl.get("area_mm2"),
            "sizes": {k: round(v, 2) for k, v in sizes.items() if v is not None},
            "status": m["status"], "name": m.get("name"),
            "confidence": round(m["confidence"], 2) if m.get("confidence") is not None else None,
            "distance": round(m["distance"], 2) if m.get("distance") is not None else None,
            "reason": m.get("reason"), "between": m.get("between"),
            "count": m.get("count", 1) if m["status"] == "identified" else 0, "group": m.get("group"),
            "candidates": m.get("candidates", [])}


def counter_rows(c):
    """Battery counter: every catalogue battery with how many were found (a group of
    touching cells counted by size adds its count), the price each and the line total."""
    found, leaking = {}, {}
    for p in c["parts"]:
        m = p["match"]
        if m["status"] == "identified":
            found[m["name"]] = found.get(m["name"], 0) + m.get("count", 1)
            if (p.get("health") or {}).get("status") == "damaged":
                leaking[m["name"]] = leaking.get(m["name"], 0) + m.get("count", 1)
    return [{"name": row["name"], "type": row["type"], "quantity": found.get(row["name"], 0),
             "leaking": leaking.get(row["name"], 0),
             "unit_price": f"{row['price']:.2f}", "line_total": f"{found.get(row['name'], 0) * row['price']:.2f}"}
            for row in c["catalogue"]]


@app.get("/api/s/{sid}/analysis")
def analysis(sid: str):
    s = get_session(sid)
    need_done(s)
    c = s.ctx
    ref = c["ref"]
    parts = sorted(c["parts"], key=lambda p: p["no"])
    rows = [part_row(p, ref is not None) for p in parts]
    counts = {"identified": 0, "ambiguous": 0, "unknown": 0}
    for r in rows:
        counts[r["status"]] += 1
    confs = [r["confidence"] for r in rows if r["status"] == "identified" and r["confidence"] is not None]
    applied = [st["id"] for st in STEPS if s.records.get(st["id"], {}).get("status") == "applied"]
    syl, ext = concepts(set(applied))
    scale = None
    if ref:
        scale = {"kind": "card", "mm_per_px": round(ref["mm_per_px"], 5),
                 "size_px": [round(v, 1) for v in ref["px"]],
                 "corner_angles": [round(a, 1) for a in ref.get("corner_angles", [])]}
    counter = counter_rows(c)
    return {
        "counter": counter, "batteries": sum(r["quantity"] for r in counter), "grand_total": c["invoice"]["grand_total"],
        "crop": c.get("crop"),
        "steps": [{"id": st["id"], "title": st["title"], "status": s.records.get(st["id"], {}).get("status", "skipped"),
                   "summary": s.records.get(st["id"], {}).get("summary", "")} for st in STEPS],
        "parts": rows, "counts": counts, "total_parts": len(rows),
        "mean_confidence": round(float(np.mean(confs)), 2) if confs else None,
        "scale": scale, "blur_var": round(c.get("blur_var", 0), 1),
        "concepts": {"syllabus": syl, "extra": ext}, "not_used": NOT_USED,
        "annotated_url": s.url(s.put(c["annotated"])),
        "health": health_summary(c),
    }


def health_summary(c):
    """Counts of OK / check / leaking batteries and the advice that goes with them."""
    n = (c.get("health") or {}).get("counts") or {"ok": 0, "check": 0, "damaged": 0}
    level = "damaged" if n["damaged"] else "check" if n["check"] else "ok"
    return {"counts": n, "level": level, "advice": HEALTH_ADVICE[level]}


@app.get("/api/s/{sid}/results")
def results(sid: str):
    s = get_session(sid)
    need_done(s)
    c = s.ctx
    h, w = c["annotated"].shape[:2]
    return {"invoice": c["invoice"], "counter": counter_rows(c), "annotated_url": s.url(s.put(c["annotated"])),
            "health": health_summary(c), "size": [w, h], "parts": [part_row(p, c["ref"] is not None) for p in sorted(c["parts"], key=lambda p: p["no"])]}


# ----------------------------------------------------------------------------- downloads


def font(size, bold=False):
    for name in (["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_pdf(s):
    c = s.ctx
    inv = c["invoice"]
    page = (1240, 1754)  # A4 at 150 dpi
    canvas = Image.new("RGB", page, "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, page[0], 130], fill=(23, 71, 184))
    d.text((page[0] // 2, 65), "Battery Counter", fill="white", font=font(54, True), anchor="mm")
    d.text((70, 160), f"Battery count  -  {s.created:%d %B %Y, %H:%M}", fill=(20, 33, 61), font=font(26, True))
    im = Image.fromarray(c["annotated"])
    k = min((page[0] - 140) / im.width, 620 / im.height)
    im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    canvas.paste(im, ((page[0] - im.width) // 2, 210))
    y = 210 + im.height + 40
    cols = [70, 720, 880, 1170]
    f, fb = font(24), font(24, True)
    d.text((cols[0], y), "Battery", font=fb, fill=(90, 107, 140))
    d.text((cols[1], y), "Count", font=fb, fill=(90, 107, 140))
    d.text((cols[2], y), "Price each", font=fb, fill=(90, 107, 140))
    d.text((cols[3], y), "Total", font=fb, fill=(90, 107, 140), anchor="ra")
    y += 38
    d.line([70, y, page[0] - 70, y], fill=(214, 226, 250), width=2)
    y += 12
    for line in counter_rows(c):
        d.text((cols[0], y), line["name"], font=f, fill=(20, 33, 61))
        d.text((cols[1], y), str(line["quantity"]), font=f, fill=(20, 33, 61))
        d.text((cols[2], y), line["unit_price"], font=f, fill=(20, 33, 61))
        d.text((cols[3], y), line["line_total"], font=f, fill=(20, 33, 61), anchor="ra")
        y += 38
    d.line([70, y, page[0] - 70, y], fill=(214, 226, 250), width=2)
    y += 14
    d.text((cols[0], y), "Grand total", font=font(28, True), fill=(18, 58, 153))
    d.text((cols[3], y), inv["grand_total"], font=font(28, True), fill=(18, 58, 153), anchor="ra")
    y += 60
    hs = health_summary(c)
    if hs["level"] != "ok":
        bad = [p for p in sorted(c["parts"], key=lambda q: q["no"]) if (p.get("health") or {}).get("status") != "ok"]
        col = (176, 20, 120) if hs["level"] == "damaged" else (160, 100, 0)
        d.rectangle([60, y - 8, page[0] - 60, y + 46 + 30 * min(len(bad), 6) + 60], outline=col, width=3)
        title = (f"WARNING: {hs['counts']['damaged']} leaking battery - do not use" if hs["counts"]["damaged"] == 1 else
                 f"WARNING: {hs['counts']['damaged']} leaking batteries - do not use" if hs["counts"]["damaged"] else
                 f"{hs['counts']['check']} battery(s) with a small deposit - check by eye")
        d.text((cols[0] + 10, y), title, font=font(26, True), fill=col)
        y += 40
        for p in bad[:6]:
            nm = p["match"].get("name") or "object"
            d.text((cols[0] + 10, y), f"Object {p['no']} ({nm}): {p['health']['reason']}"[:92], font=font(21), fill=(60, 70, 100))
            y += 30
        for line in [hs["advice"][i:i + 100] for i in range(0, min(len(hs["advice"]), 200), 100)]:
            d.text((cols[0] + 10, y), line, font=font(19), fill=(60, 70, 100))
            y += 26
        y += 40
    if inv["not_charged"]:
        d.text((cols[0], y), "Not counted (could not be named, check by hand):", font=fb, fill=(154, 52, 18))
        y += 36
        for item in inv["not_charged"][:14]:
            no = next((p["no"] for p in c["parts"] if p["label"] == item["label"]), "?")
            txt = f"Object {no}: {item['status']} - {item['reason']}"
            if item.get("between"):
                txt += f" ({item['between'][0]} or {item['between'][1]})"
            d.text((cols[0], y), txt[:95], font=font(21), fill=(60, 70, 100))
            y += 30
    d.text((70, page[1] - 70), "Counted and measured by classical image processing, with a credit card for scale.",
           font=font(19), fill=(90, 107, 140))
    buf = io.BytesIO()
    canvas.save(buf, "PDF", resolution=150)
    return buf.getvalue()


def make_csv(s):
    c = s.ctx
    inv = c["invoice"]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Battery Counter", f"{s.created:%Y-%m-%d %H:%M}"])
    w.writerow([])
    w.writerow(["battery", "count", "price_each", "total", "leaking"])
    rows = counter_rows(c)
    for line in rows:
        w.writerow([line["name"], line["quantity"], line["unit_price"], line["line_total"], line["leaking"]])
    w.writerow(["GRAND TOTAL", sum(r["quantity"] for r in rows), "", inv["grand_total"], sum(r["leaking"] for r in rows)])
    w.writerow([])
    w.writerow(["no", "shape", "status", "name", "count", "sizes_mm", "confidence", "reason", "condition", "leak_mm2", "condition_note"])
    for p in sorted(c["parts"], key=lambda p: p["no"]):
        r = part_row(p, c["ref"] is not None)
        cond = {"ok": "OK", "check": "CHECK", "damaged": "LEAKING"}.get(r["health"], "OK")
        w.writerow([r["no"], r["type"], r["status"], r["name"] or "", r["count"],
                    "; ".join(f"{k} {v}" for k, v in r["sizes"].items()),
                    r["confidence"] if r["confidence"] is not None else "", r["reason"] or r["group"] or "",
                    cond, r["leak_mm2"] if r["leak_mm2"] is not None else "", r["health_reason"] if r["health"] != "ok" else ""])
    return out.getvalue().encode("utf-8-sig")


@app.get("/api/s/{sid}/download/{kind}")
def download(sid: str, kind: str):
    s = get_session(sid)
    need_done(s)
    stamp = f"{s.created:%Y%m%d-%H%M}"
    if kind == "jpg":
        buf = io.BytesIO()
        Image.fromarray(s.ctx["annotated"]).save(buf, "JPEG", quality=93)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="batteries-{stamp}.jpg"'})
    if kind == "pdf":
        return Response(make_pdf(s), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="batteries-{stamp}.pdf"'})
    if kind == "csv":
        return Response(make_csv(s), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="batteries-{stamp}.csv"'})
    raise HTTPException(404, "Unknown format.")


# ----------------------------------------------------------------------------- leakage check page


@app.post("/api/leakcheck")
def leakcheck(file: UploadFile = File(...)):
    """The separate Leakage check page: the same steps as the counter, up to and including
    the leak check, on one photo. A card is optional: without one the deposit is judged as a
    share of each battery's own area instead of in mm^2."""
    arr, info = read_image(file.file.read(MAX_BYTES + 1))
    s = Session(arr, info)
    with LOCK:
        SESSIONS[s.id] = s
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    c = s.ctx
    t0 = time.time()
    hres = None
    try:
        for st in STEPS:
            r = st["fn"](c)
            if st["id"] == "health":
                hres = r
                break
    except Exception as exc:
        raise HTTPException(500, f"The leak check failed ({type(exc).__name__}: {exc}).")
    parts = sorted(c.get("parts", []), key=lambda p: p.get("no", 0))
    if hres is None or hres["status"] != "applied":
        return {"found": 0, "message": "No battery was found in the photo. Put the batteries on plain white paper and try again.",
                "original_url": s.url(s.original_key)}
    rows = []
    for p in parts:
        hl = p["health"]
        m = p["match"]
        rows.append({"no": p.get("no"), "name": m.get("name") if m["status"] == "identified" else None,
                     "shape": TYPE_NAME.get(p["type"], p["type"]), "status": hl["status"], "reason": hl["reason"],
                     "area_mm2": hl["area_mm2"], "frac": hl["frac"], "kinds": hl["kinds"]})
    hs = health_summary(c)
    return {"found": len(parts), "seconds": round(time.time() - t0, 1), "has_scale": c.get("ref") is not None,
            "original_url": s.url(s.original_key),
            "annotated_url": s.url(s.put(c["annotated"])),
            "leak_url": s.url(s.put(hres["after"])),
            "aux": [{"label": lbl, "url": s.url(s.put(im, png=True))} for lbl, im in hres.get("aux", [])],
            "rows": rows, "summary": hres["summary"], "facts": hres.get("facts", []), **hs}


@app.get("/api/leak_report")
def leak_report():
    f = ROOT / "data" / "tests" / "leak_report.json"
    if not f.exists():
        raise HTTPException(404, "Run tools/test_leakage.py first.")
    import json
    return json.loads(f.read_text())


@app.get("/api/samples")
def samples():
    """The drawn battery test photos, offered on the Upload page so the site can be tried at once."""
    out = []
    for p in sorted((ROOT / "data" / "tests").glob("battery_*.jpg")):
        out.append({"name": p.name, "url": f"/samples/{p.name}"})
    return out


@app.get("/samples/{name}")
def sample_file(name: str):
    p = (ROOT / "data" / "tests" / name).resolve()
    if p.parent != (ROOT / "data" / "tests").resolve() or not p.is_file():
        raise HTTPException(404, "No such sample.")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
