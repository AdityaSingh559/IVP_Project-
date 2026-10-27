"""Battery count and total: count x catalogue price per battery, and the grand total."""

from decimal import ROUND_HALF_UP, Decimal


def money(x):
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def build(parts, catalogue):
    """parts: measured objects with p['match']. Ambiguous and unknown objects are listed
    separately and are not counted in the total."""
    qty = {}
    for p in parts:
        m = p.get("match") or {}
        if m.get("status") == "identified":
            # a group of touching cells counted by its size counts as several
            qty[m["row"]] = qty.get(m["row"], 0) + m.get("count", 1)
    lines, total = [], Decimal("0.00")
    for i in sorted(qty, key=lambda i: catalogue[i]["name"]):
        r = catalogue[i]
        unit = money(r["price"])
        line = unit * qty[i]
        total += line
        lines.append({"name": r["name"], "quantity": qty[i], "unit_price": str(unit),
                      "line_total": str(line)})
    flagged = [{"label": p["label"], "status": p["match"]["status"],
                "reason": p["match"].get("reason", ""), "between": p["match"].get("between")}
               for p in parts if (p.get("match") or {}).get("status") in ("ambiguous", "unknown")]
    return {"lines": lines, "grand_total": str(total), "not_charged": flagged}
