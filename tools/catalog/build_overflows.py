"""Research YAML -> reefflow/data/overflow_products.yaml (curated)."""
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OUT = REPO / "reefflow/data/overflow_products.yaml"
d = yaml.safe_load((HERE / "research" / "overflows.yaml").read_text(encoding="utf-8"))

SKIP = {"glass_holes_700", "glass_holes_1500", "reef_savvy_ghost_retail_2016"}
HOB_NOTE = ("Published depth is the whole unit front to back (in-tank box plus the part behind the glass); "
            "the in-tank depth is not published, so picking it keeps your depth. Measure the in-tank box.")


def clean_flow(s):
    """'400-2500 GPH' -> '2500 GPH'; '800+ GPH' -> '800 GPH'; '2000 L/h (530 GPH)' -> '2000 L/h'."""
    if not s:
        return None
    s = str(s).split("(")[0].strip()
    m = re.match(r"^(?:up to\s+)?(?:[\d.,]+\s*-\s*)?([\d.,]+)\+?\s*(GPH|L/h)$", s, re.I)
    if not m:
        raise ValueError(f"cannot read rated flow {s!r}")
    return f"{m.group(1).replace(',', '')} {m.group(2)}"


def mixed(v):
    """'7 1/4 in' -> '7.25 in' (the units parser reads '1/4 in' but not mixed numbers)."""
    m = re.match(r"^(\d+) (\d+)/(\d+) (\w+)$", str(v))
    return f"{int(m.group(1)) + int(m.group(2)) / int(m.group(3)):g} {m.group(4)}" if m else v


def status(s):
    s = str(s or "")
    if s.startswith("current"):
        return "current"
    if s.startswith("discontinued"):
        return "discontinued"
    return "unknown"


prods = {}
for p in d["products"]:
    if p["id"] in SKIP:
        continue
    box = {k: mixed(v) for k, v in (p.get("box") or {}).items() if v}
    notes = [str(p.get("notes") or "").strip()]
    if p.get("mount") == "hang_on_back":
        box.pop("depth", None)
        box.pop("height", None)
        notes.insert(0, HOB_NOTE)
    st = status(p.get("status"))
    if st == "unknown":
        notes.insert(0, f"Status unknown: {p.get('status')}.")
    e = {"brand": p["brand"], "model": p["model"], "mount": p.get("mount"),
         "box": box or None, "rated_flow": clean_flow(p.get("rated_flow")),
         "rated_flow_basis": (f"published as '{p['rated_flow']}'. " if p.get("rated_flow") else "") + str(p.get("rated_flow_basis") or "not published").strip(),
         "design": p.get("design"), "drains": p.get("drains"),
         "status": st, "checked": str(p.get("checked") or "2026-11-12"),
         "sources": [s for s in (p.get("sources") or []) if str(s).startswith("http")],
         "notes": " ".join(n for n in notes if n)}
    if p["brand"].startswith("Red Sea"):
        e["rated_flow_basis"] = ("Red Sea's recommended return-pump flow for this model (G2 manual), not a tested "
                                 "drain capacity. " + e["rated_flow_basis"])
        e["notes"] = ("Red Sea does not publish the overflow box size: picking this model keeps your box size and "
                      "sets only the capacity. " + e["notes"])
    prods[p["id"]] = {k: v for k, v in e.items() if v is not None}

dc = []
for r in d["drain_capacity"]:
    src = str(r.get("source") or "")
    if "aquaculator" in src.lower() or "BeanAnimal" in src or not (r.get("siphon") or r.get("gravity")):
        continue
    url = re.search(r"https?://\S+", src)
    dc.append({"pipe": r["pipe"], "siphon": r.get("siphon"), "gravity": r.get("gravity"),
               "source": url.group(0).rstrip(",)") if url else src,
               "source_name": src.split(",")[0].split(" - ")[0].strip(),
               "notes": r.get("notes")})

head = """# ReefFlow overflow catalogue (D-844) and drain capacities (D-846).
#
# Picking a product sets the overflow box size (width along the glass, depth out from the
# glass) and its rated capacity. The capacity is only used to WARN when the return flow is
# higher; the flow model does not limit the overflow. Values are as published by the
# maker (or a retailer copying the maker's table, as noted) and were checked 2026-11-12.
# The weir crest depth is not published for any of these; keep your measured value.
#
# drain_capacity: commonly quoted single-drain capacities by pipe size. The sources
# disagree by up to 2x (measured test vs rules of thumb); the measured test is the
# conservative one. BeanAnimal's write-up gives no per-pipe table (one tank: 3 x 1 in
# bulkheads, 2000 GPH silent).
"""
OUT.write_text(head + yaml.safe_dump({"products": prods, "drain_capacity": dc}, sort_keys=False,
                                     allow_unicode=True, width=110), encoding="utf-8")
print(len(prods), "products,", len(dc), "drain rows")
for k, v in prods.items():
    print(f"  {k:32s} {v.get('rated_flow', '-'):>10s}  {v.get('box')}  {v['status']}")
