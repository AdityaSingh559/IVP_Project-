"""Battery catalogue: CSV loading and validation.

Columns: name, type, size_mm, second_mm, price (other columns are ignored).
  cylinder  a cell lying on its side (AA, AAA): size_mm = length, second_mm = diameter
  button    a coin or button cell lying flat (CR2016, LR44): size_mm = diameter
"""

import csv
import io

COLUMNS = ["name", "type", "size_mm", "second_mm", "price"]
TYPES = {"cylinder", "button"}


def _num(s):
    s = (s or "").strip()
    return float(s) if s else None


def parse_csv(text):
    """Parse catalogue CSV text. Returns (rows, errors)."""
    rows, errors = [], []
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        return [], ["missing columns: " + ", ".join(missing)]
    for i, r in enumerate(reader, start=2):
        try:
            row = {"name": r["name"].strip(), "type": r["type"].strip().lower(),
                   "size_mm": _num(r["size_mm"]), "second_mm": _num(r["second_mm"]),
                   "price": _num(r["price"])}
        except ValueError as e:
            errors.append(f"line {i}: {e}")
            continue
        if row["type"] not in TYPES:
            errors.append(f"line {i}: type must be cylinder or button, not '{row['type']}'")
            continue
        if row["size_mm"] is None:
            errors.append(f"line {i}: size_mm is missing")
            continue
        if row["type"] == "cylinder" and row["second_mm"] is None:
            errors.append(f"line {i}: a cylinder needs its diameter in second_mm")
            continue
        if row["type"] == "button":
            row["second_mm"] = None
        if row["price"] is None:
            errors.append(f"line {i}: price is missing")
            continue
        rows.append(row)
    return rows, errors


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return parse_csv(f.read())
