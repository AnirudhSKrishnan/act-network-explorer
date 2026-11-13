"""Run the whole pipeline from cached raw data and write the summary.

Run with::

    python -m act_network.build

Steps: clean -> report -> service_areas -> webmap -> static_map -> summary.
It does NOT fetch: if the cached raw files are missing it tells you to run
``python -m act_network.fetch`` first and exits with status 1.

``outputs/summary.json`` / ``outputs/summary.md`` are the single source of
numbers for the resume. Every number is computed from the processed data and
the check outputs; none is typed by hand.

Data provenance: public OpenStreetMap data (c) OpenStreetMap contributors
(ODbL); suburbs (c) ABS (CC BY 4.0). NOT Evoenergy network data.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from act_network import clean, fetch, report, service_areas, static_map, webmap
from act_network.webmap import DISCLAIMER

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
SUMMARY_JSON = OUTPUTS_DIR / "summary.json"
SUMMARY_MD = OUTPUTS_DIR / "summary.md"
FEEDER_JSON = OUTPUTS_DIR / "feeder_results.json"

BAND_ORDER = ["330 kV", "132 kV", "66 kV", "22 kV", "11 kV", "other", "LV (415 V)", "unknown"]
CLASS_ORDER = ["transmission", "distribution", "unknown"]
SUB_CLASS_ORDER = ["transmission", "zone", "distribution"]
GROUP_LABELS = {"likely_error": "likely error", "to_check": "to check", "tag_gap": "tag gap"}
TOP_N_SUBURBS = 5


# ---------------------------------------------------------------------------
# pure summary maths
# ---------------------------------------------------------------------------


def _km(series: pd.Series) -> float:
    return round(float(series.sum()) / 1000.0, 1)


def _ordered(d: dict, order: list[str]) -> dict:
    return {k: d[k] for k in order if k in d} | {k: v for k, v in d.items() if k not in order}


def summarise_lines(lines: pd.DataFrame) -> dict:
    """km of line (from ``length_m``) by voltage band, by line_class, and underground."""
    km_band = (lines.groupby(lines["voltage_band"].fillna("unknown"))["length_m"].sum() / 1000).round(1)
    km_class = (lines.groupby(lines["line_class"].fillna("unknown"))["length_m"].sum() / 1000).round(1)
    ug = lines["is_underground"].fillna(False).astype(bool)
    return {
        "n_line_segments": int(len(lines)),
        "total_km": _km(lines["length_m"]),
        "km_by_voltage_band": _ordered(km_band.to_dict(), BAND_ORDER),
        "km_by_line_class": _ordered(km_class.to_dict(), CLASS_ORDER),
        "km_underground": _km(lines.loc[ug, "length_m"]),
        "n_underground_segments": int(ug.sum()),
    }


def summarise_substations(subs: pd.DataFrame) -> dict:
    counts = subs["sub_class"].fillna("distribution").value_counts().to_dict()
    return {"total": int(len(subs)), "by_sub_class": _ordered({k: int(v) for k, v in counts.items()}, SUB_CLASS_ORDER)}


def top_furthest_suburbs(dists: pd.DataFrame, n: int = TOP_N_SUBURBS) -> list[dict]:
    """The n real (non "ACT Remainder") suburbs furthest from their nearest zone substation."""
    real = dists[~dists["is_remainder"].astype(bool)].sort_values("distance_km", ascending=False).head(n)
    return [
        {"rank": i + 1, "suburb": r["suburb_name"], "distance_km": round(float(r["distance_km"]), 1),
         "nearest_substation": r["nearest_substation"]}
        for i, (_, r) in enumerate(real.iterrows())
    ]


def quality_counts(quality_summary: dict) -> dict:
    """Issue counts by group and per check, read straight from quality_summary.json."""
    iss = quality_summary["issues"]
    return {
        "total_rows": iss["total_rows"],
        "by_group": dict(iss["by_group"]),
        "by_check": dict(iss["by_type"]),
        "by_check_and_group": dict(iss["by_type_and_group"]),
        "distinct_osm_objects_flagged": iss.get("distinct_osm_objects_flagged"),
    }


def build_summary(lines, substations, n_service_areas: int, dists, quality_summary: dict,
                  today: dt.date | None = None) -> dict:
    return {
        "generated": (today or dt.date.today()).isoformat(),
        "data_note": DISCLAIMER,
        "lines": summarise_lines(lines),
        "substations": summarise_substations(substations),
        "zone_service_areas": int(n_service_areas),
        "quality_issues": quality_counts(quality_summary),
        "top_furthest_suburbs": top_furthest_suburbs(dists),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _flatten(d: dict, prefix: str = "", depth: int = 2) -> list[tuple[str, object]]:
    items = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth > 1:
            items += _flatten(v, key + " / ", depth - 1)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            items.append((key, v))
    return items


def render_markdown(s: dict, feeder: dict | None = None) -> str:
    ln, qi = s["lines"], s["quality_issues"]
    md = [
        "# ACT electricity network from OpenStreetMap: summary of numbers",
        "",
        f"Generated: {s['generated']}",
        "",
        f"> {s['data_note']}",
        "",
        "Every figure below is computed by `python -m act_network.build` from the processed data and check "
        "outputs. Line lengths are the sum of mapped OSM way lengths (EPSG:7855); overlapping or duplicated "
        "ways are counted as mapped.",
        "",
        "## Line length",
        "",
        f"Total: **{ln['total_km']:,.1f} km** in {ln['n_line_segments']:,} mapped segments; "
        f"**{ln['km_underground']:,.1f} km** underground ({ln['n_underground_segments']:,} segments).",
        "",
        "By voltage band:",
        "",
        _table(["Voltage band", "km"], [[k, f"{v:,.1f}"] for k, v in ln["km_by_voltage_band"].items()]),
        "",
        "By line class (transmission = 66 kV and above; unknown = no usable voltage tag):",
        "",
        _table(["Line class", "km"], [[k, f"{v:,.1f}"] for k, v in ln["km_by_line_class"].items()]),
        "",
        "## Substations",
        "",
        _table(["Class", "Count"], [[k, v] for k, v in s["substations"]["by_sub_class"].items()]
               + [["**total**", f"**{s['substations']['total']}**"]]),
        "",
        f"Rough nearest-substation areas (geometric estimate, not feeder boundaries): "
        f"**{s['zone_service_areas']}** (one per zone substation after merging duplicates).",
        "",
        "## Data-quality issues",
        "",
        f"**{qi['total_rows']:,}** issue rows flagged"
        + (f" across {qi['distinct_osm_objects_flagged']:,} distinct OSM objects." if qi.get("distinct_osm_objects_flagged") else "."),
        "",
        "By group:",
        "",
        _table(["Group", "Issues"], [[GROUP_LABELS.get(k, k), f"{v:,}"] for k, v in qi["by_group"].items()]),
        "",
        "By check:",
        "",
        _table(["Check", "Issues"], [[k.replace("_", " "), f"{v:,}"] for k, v in qi["by_check"].items()]),
        "",
        "## Furthest suburbs from the nearest zone substation",
        "",
        "Straight-line distance from a suburb's representative point to the nearest zone substation "
        "(not along the network).",
        "",
        _table(["#", "Suburb", "Distance (km)", "Nearest zone substation"],
               [[r["rank"], r["suburb"], f"{r['distance_km']:.1f}", r["nearest_substation"]]
                for r in s["top_furthest_suburbs"]]),
        "",
    ]
    # OPTIONAL HOOK: feeder load-flow results, added by a later step (feeder.py writes
    # outputs/feeder_results.json). Omitted entirely if that file does not exist.
    if feeder:
        md += ["## Toy 11 kV feeder load flow (pandapower)", "",
               "Toy model: assumed loads and a standard library cable type, NOT Evoenergy data. "
               "Assumptions are listed with the feeder step's output.", ""]
        md += [f"- {k}: {v}" for k, v in _flatten(feeder)]
        md += [""]
    return "\n".join(md)


def load_feeder(path: Path = FEEDER_JSON) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

RAW_FILES = [fetch.NETWORK_PATH, fetch.SUPPORTS_PATH, fetch.BOUNDARY_PATH, fetch.SUBURBS_PATH]


def missing_raw_files() -> list[Path]:
    return [p for p in RAW_FILES if not Path(p).exists()]


def write_summary() -> dict:
    lines = gpd.read_file(clean.PROCESSED_PATH, layer=clean.LINES_LAYER)
    subs = gpd.read_file(clean.PROCESSED_PATH, layer=clean.SUBSTATIONS_LAYER)
    areas = gpd.read_file(service_areas.SERVICE_AREAS_PATH, layer=service_areas.AREAS_LAYER)
    dists = gpd.read_file(service_areas.SERVICE_AREAS_PATH, layer=service_areas.SUBURB_DISTANCES_LAYER)
    quality = json.loads(report.SUMMARY_JSON.read_text())
    summary = build_summary(lines, subs, len(areas), dists, quality)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    SUMMARY_MD.write_text(render_markdown(summary, load_feeder()) + "\n")
    print(f"Wrote {SUMMARY_JSON} and {SUMMARY_MD}")
    return summary


def _step(name: str, fn) -> None:
    print(f"\n=== {name} ===", flush=True)
    rc = fn()
    if rc not in (None, 0):
        raise SystemExit(f"{name} failed (exit status {rc})")


def main() -> int:
    missing = missing_raw_files()
    if missing:
        print("Cached raw data is missing:\n  " + "\n  ".join(str(p) for p in missing))
        print("Run `python -m act_network.fetch` first (needs internet), then re-run the build.")
        return 1
    _step("clean", clean.main)
    _step("report (topology + quality checks)", report.main)
    _step("service_areas", service_areas.main)
    _step("webmap", webmap.main)
    _step("static_map", static_map.main)
    print("\n=== summary ===")
    write_summary()
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
