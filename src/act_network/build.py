"""Run the whole pipeline from cached raw data and write the summary.

Run with::

    python -m act_network.build

Steps: clean -> report -> service_areas -> feeder -> webmap -> static_map -> summary.
It does NOT fetch: if the cached raw files are missing it tells you to run
``python -m act_network.fetch`` first and exits with status 1.

``outputs/summary.json`` / ``outputs/summary.md`` are the single source of
numbers for the resume. Every number is computed from the processed data and
the check outputs; none is typed by hand.

Data provenance: public OpenStreetMap data (c) OpenStreetMap contributors
(ODbL); suburbs (c) ABS (CC BY 4.0). NOT official network operator data.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from act_network import clean, feeder, fetch, report, service_areas, static_map, webmap
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


def _km(series: pd.Series, nd: int = 1) -> float:
    return round(float(series.sum()) / 1000.0, nd)


def _ordered(d: dict, order: list[str]) -> dict:
    return {k: d[k] for k in order if k in d} | {k: v for k, v in d.items() if k not in order}


def _km_by(lines: pd.DataFrame, key: pd.Series, col: str) -> dict:
    return (lines.groupby(key)[col].sum() / 1000).round(1).to_dict()


def summarise_lines(lines: pd.DataFrame) -> dict:
    """km of line by voltage band, line_class and underground.

    Headline = ``length_in_act_m`` (length clipped to the ACT boundary). osmnx returns
    whole OSM ways that cross the border, so ``length_m`` ("whole ways") also counts the
    NSW sections of those ways; it is reported as a secondary figure only.
    """
    if "length_in_act_m" not in lines.columns:
        raise ValueError("lines needs a 'length_in_act_m' column (re-run `python -m act_network.clean`)")
    act, whole = lines["length_in_act_m"], lines["length_m"]
    band = lines["voltage_band"].fillna("unknown")
    cls = lines["line_class"].fillna("unknown")
    ug = lines["is_underground"].fillna(False).astype(bool)
    internal = (lines["is_substation_internal"].fillna(False).astype(bool)
                if "is_substation_internal" in lines.columns else pd.Series(False, index=lines.index))
    crossing = (act > 0) & (act < whole - 1.0)  # >1 m of the way lies outside the ACT
    tx = cls == "transmission"
    return {
        "n_line_segments": int(len(lines)),
        "n_segments_crossing_act_boundary": int(crossing.sum()),
        "n_segments_wholly_outside_act": int((act <= 0).sum()),
        "total_km": _km(act),
        "total_km_whole_ways": _km(whole),
        "km_by_voltage_band": _ordered(_km_by(lines, band, "length_in_act_m"), BAND_ORDER),
        "km_by_voltage_band_whole_ways": _ordered(_km_by(lines, band, "length_m"), BAND_ORDER),
        "km_by_line_class": _ordered(_km_by(lines, cls, "length_in_act_m"), CLASS_ORDER),
        "km_by_line_class_whole_ways": _ordered(_km_by(lines, cls, "length_m"), CLASS_ORDER),
        "km_underground": _km(act[ug]),
        "km_underground_whole_ways": _km(whole[ug]),
        "n_underground_segments": int(ug.sum()),
        "n_substation_bay_busbar_ways": int(internal.sum()),
        "km_substation_bays_busbars": _km(act[internal], 2),
        "km_substation_bays_busbars_whole_ways": _km(whole[internal], 2),
        "km_substation_bays_busbars_by_line_class": _ordered(
            (lines[internal].groupby(cls[internal])["length_in_act_m"].sum() / 1000).round(2).to_dict(), CLASS_ORDER),
        "km_transmission_excl_bays_busbars": _km(act[tx & ~internal]),
    }


def summarise_substations(subs: pd.DataFrame, n_duplicate_features: int | None = None) -> dict:
    """Counts of OSM substation *features* by class (features are not sites).

    ``zone_sites_by_name`` counts distinct zone-substation names, i.e. zone features after
    merging same-name duplicates (the figure used for the rough nearest-substation areas).
    ``n_duplicate_features_flagged`` is the number of substation rows the quality checks flagged
    as a duplicate or nested footprint of another feature.
    """
    cls = subs["sub_class"].fillna("distribution")
    counts = cls.value_counts().to_dict()
    zone_names = subs.loc[cls == "zone", "name"].dropna().astype(str).str.strip().str.lower()
    n_zone_unnamed = int(((cls == "zone") & subs["name"].isna()).sum())
    zn = subs.loc[cls == "zone", "name"].dropna().astype(str).str.strip()
    out = {
        "zone_duplicate_names": sorted(zn[zn.str.lower().duplicated(keep=False)].unique()),
        "total": int(len(subs)),
        "by_sub_class": _ordered({k: int(v) for k, v in counts.items()}, SUB_CLASS_ORDER),
        "zone_sites_by_name": int(zone_names.nunique()) + n_zone_unnamed,
        "n_unnamed": int(subs["name"].isna().sum()),
    }
    if "in_act" in subs.columns:
        out["n_in_act"] = int(subs["in_act"].fillna(False).astype(bool).sum())
    if n_duplicate_features is not None:
        out["n_duplicate_features_flagged"] = int(n_duplicate_features)
    return out


def top_furthest_suburbs(dists: pd.DataFrame, n: int = TOP_N_SUBURBS) -> list[dict]:
    """The n real (non "ACT Remainder") suburbs furthest from their nearest zone substation."""
    real = dists[~dists["is_remainder"].astype(bool)].sort_values("distance_km", ascending=False).head(n)
    return [
        {"rank": i + 1, "suburb": r["suburb_name"], "distance_km": round(float(r["distance_km"]), 1),
         "nearest_substation": r["nearest_substation"]}
        for i, (_, r) in enumerate(real.iterrows())
    ]


def quality_counts(quality_summary: dict) -> dict:
    """Issue counts by group and per check, read straight from quality_summary.json.

    "likely error" is the group name used by the checks: rows the checks flagged as probable
    mapping errors, not verified errors. They are split into rows inside substation sites and
    elsewhere. Total rows is a row count, not an error count.
    """
    iss = quality_summary["issues"]
    chk = quality_summary.get("checks", {})
    ov = chk.get("overlapping_lines", {})
    by_group = dict(iss["by_group"])
    likely = by_group.get("likely_error", 0)
    outside = iss.get("likely_error_outside_substations")
    return {
        "total_rows": iss["total_rows"],
        "by_group": by_group,
        "by_check": dict(iss["by_type"]),
        "by_check_and_group": dict(iss["by_type_and_group"]),
        "distinct_osm_objects_flagged": iss.get("distinct_osm_objects_flagged"),
        "likely_error_outside_substations": outside,
        "likely_error_inside_substations": None if outside is None else int(likely - outside),
        "overlapping_lines_inside_substations": ov.get("inside_substations"),
        "overlapping_lines_substation_sites": ov.get("substation_sites"),
        "overlapping_lines_total": ov.get("count"),
        "duplicate_substations_by_kind": dict(chk.get("duplicate_substations", {})),
    }


def transformer_counts(quality_summary: dict) -> dict:
    net = quality_summary.get("network", {})
    return {"total": net.get("n_transformers"), "by_source": dict(net.get("n_transformers_by_source", {}))}


def _r(v, nd: int):
    return None if v is None else round(float(v), nd)


def summarise_feeder(f: dict | None) -> dict | None:
    """The numbers worth quoting from outputs/feeder_results.json (None if there are none)."""
    if not f:
        return None
    path, model, res = f.get("path", {}), f.get("model", {}), f.get("results", {})
    lim, ref = f.get("limits", {}), f.get("reference_limits", {})
    if not res.get("base") or not path:
        return None

    def case(r: dict) -> dict:
        return {
            "load_multiplier": r.get("load_multiplier"),
            "total_load_kw": _r(r.get("total_load_kw"), 0),
            "end_voltage_pu": _r(r.get("end_vm_pu"), 3),
            "end_voltage_drop_pct": _r(r.get("end_voltage_drop_pct"), 2),
            "max_line_loading_pct": _r(r.get("max_line_loading_pct"), 1),
            "total_losses_kw": _r(r.get("total_losses_kw"), 1),
            "converged": r.get("converged"),
        }

    n_way_ids = path.get("osm_way_ids") or []
    return {
        "substation": path.get("substation_name"),
        "substation_osm_id": path.get("substation_osm_id"),
        "fallback_path": path.get("fallback"),
        "path_length_km": _r(None if path.get("total_length_m") is None else path["total_length_m"] / 1000, 2),
        "n_osm_ways": len(n_way_ids),
        "n_buses": path.get("n_buses"),
        "n_model_sections": path.get("n_model_sections"),
        "reachable_11kv_km": _r(None if path.get("reachable_11kv_length_m") is None
                                else path["reachable_11kv_length_m"] / 1000, 2),
        "n_zone_substations": path.get("n_zone_substations"),
        "n_zone_substations_with_11kv": path.get("n_zone_substations_with_11kv"),
        "cable_std_type": model.get("std_type"),
        "n_loads": model.get("n_loads"),
        "load_kw_each": model.get("load_kw"),
        "power_factor": model.get("power_factor"),
        "source_vm_pu": model.get("source_vm_pu"),
        "base": case(res["base"]),
        "doubled": case(res.get("doubled", {})) if res.get("doubled") else None,
        "voltage_limit_multiplier": _r(lim.get("voltage_multiplier"), 2),
        "thermal_limit_multiplier": _r(lim.get("thermal_multiplier"), 2),
        "binding_limit": lim.get("binding"),
        "v_min_pu_reference": ref.get("v_min_pu"),
        "loading_limit_pct_reference": ref.get("loading_limit_pct"),
    }


def build_summary(lines, substations, n_service_areas: int, dists, quality_summary: dict,
                  feeder_results: dict | None = None, duplicate_substation_notes: list[str] | None = None,
                  today: dt.date | None = None) -> dict:
    qc = quality_counts(quality_summary)
    subs = summarise_substations(substations, sum(qc["duplicate_substations_by_kind"].values()) or None)
    subs["flagged_duplicates"] = list(duplicate_substation_notes or [])
    return {
        "generated": (today or dt.date.today()).isoformat(),
        "data_note": DISCLAIMER,
        "lines": summarise_lines(lines),
        "substations": subs,
        "zone_service_areas": int(n_service_areas),
        "transformers": transformer_counts(quality_summary),
        "quality_issues": qc,
        "top_furthest_suburbs": top_furthest_suburbs(dists),
        "feeder": summarise_feeder(feeder_results),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fmt(v, spec: str = "", suffix: str = "") -> str:
    return "n/a" if v is None else f"{v:{spec}}{suffix}"


def render_markdown(s: dict) -> str:
    ln, qi, sub, tx = s["lines"], s["quality_issues"], s["substations"], s["transformers"]
    bands, bands_w = ln["km_by_voltage_band"], ln["km_by_voltage_band_whole_ways"]
    cls, cls_w = ln["km_by_line_class"], ln["km_by_line_class_whole_ways"]
    whole_hdr = "Whole mapped ways incl. NSW sections (km)"
    md = [
        "# ACT electricity network from OpenStreetMap: summary of numbers",
        "",
        f"Generated: {s['generated']}",
        "",
        f"> {s['data_note']}",
        "",
        "Every figure below is computed by `python -m act_network.build` from the processed data and check "
        "outputs. Nothing here is official network operator data, and OSM coverage is incomplete, so these are counts of what "
        "is mapped in OpenStreetMap, not of the real network.",
        "",
        "## Line length",
        "",
        f"**Headline: {ln['total_km']:,.1f} km of mapped line inside the ACT** "
        f"({ln['km_underground']:,.1f} km of it underground), from {ln['n_line_segments']:,} mapped OSM "
        "segments. Lengths are measured in EPSG:7855 and clipped to the ACT boundary "
        "(`length_in_act_m`). Overlapping or duplicated ways are counted as mapped.",
        "",
        f"Why two columns: OpenStreetMap queries return whole ways, and {ln['n_segments_crossing_act_boundary']:,} "
        f"of the segments cross the ACT border"
        + (f" (a further {ln['n_segments_wholly_outside_act']:,} lie wholly outside it)" if ln["n_segments_wholly_outside_act"] else "")
        + f". Summing whole-way lengths gives {ln['total_km_whole_ways']:,.1f} km "
        f"({ln['km_underground_whole_ways']:,.1f} km underground), which includes NSW sections. "
        "Use the in-ACT figures.",
        "",
        "By voltage band:",
        "",
        _table(["Voltage band", "km inside the ACT", whole_hdr],
               [[k, f"{v:,.1f}", f"{bands_w.get(k, 0.0):,.1f}"] for k, v in bands.items()]
               + [["**total**", f"**{ln['total_km']:,.1f}**", f"{ln['total_km_whole_ways']:,.1f}"]]),
        "",
        "By line class (transmission = 66 kV and above; unknown = no usable voltage tag):",
        "",
        _table(["Line class", "km inside the ACT", whole_hdr],
               [[k, f"{v:,.1f}", f"{cls_w.get(k, 0.0):,.1f}"] for k, v in cls.items()]),
        "",
        "Underground (power=cable or location=underground): "
        f"**{ln['km_underground']:,.1f} km** inside the ACT ({ln['km_underground_whole_ways']:,.1f} km whole "
        f"mapped ways; {ln['n_underground_segments']:,} segments).",
        "",
    ]
    if ln["n_substation_bay_busbar_ways"]:
        by_cls = ", ".join(f"{v:,.2f} km {k}" for k, v in ln["km_substation_bays_busbars_by_line_class"].items() if v)
        md += [
            f"Substation bays and busbars: {ln['n_substation_bay_busbar_ways']:,} ways tagged `line=bay` / "
            f"`line=busbar` ({ln['km_substation_bays_busbars']:,.2f} km inside the ACT, {ln['km_substation_bays_busbars_whole_ways']:,.2f} km whole ways; {by_cls} inside the ACT). They are "
            "conductors inside substation sites, not network lines, but they are included in the totals and "
            "bands above. Excluding them, transmission is "
            f"{ln['km_transmission_excl_bays_busbars']:,.1f} km inside the ACT.",
            "",
        ]
    zone_dups = ", ".join(f"\"{d}\"" for d in sub.get("zone_duplicate_names", []))
    md += [
        "## Substations",
        "",
        "Counts of OSM substation **features**, by class. Features are not the same as sites: the same site can "
        "be mapped more than once.",
        "",
        _table(["Class", "OSM features"], [[k, v] for k, v in sub["by_sub_class"].items()]
               + [["**total**", f"**{sub['total']}**"]]),
        "",
        f"- Zone substations: {sub['by_sub_class'].get('zone', 0)} features, **{sub['zone_sites_by_name']} "
        f"distinct sites** after merging same-name duplicates"
        + (f" ({zone_dups} is mapped twice)." if zone_dups else ".")
        + f" The rough nearest-substation areas ({s['zone_service_areas']}) use one seed per site. They are a "
        "geometric estimate, not feeder boundaries.",
    ]
    if sub.get("flagged_duplicates"):
        md += [f"- The quality checks also flagged {sub.get('n_duplicate_features_flagged', 0)} substation "
               "features as a duplicate or nested footprint of another feature:"]
        md += [f"  - {d}" for d in sub["flagged_duplicates"]]
    if sub.get("n_unnamed"):
        md += [f"- {sub['n_unnamed']} substation features have no name tag."]
    md += [
        "",
        "Transformers (OSM point features, not a count of installed transformers): "
        f"**{_fmt(tx['total'])}** total"
        + (" (" + ", ".join(f"{v} from {k}" for k, v in tx["by_source"].items()) + ")." if tx["by_source"] else ".")
        + " `power=transformer` nodes are mapped devices; \"pole transformer tag\" means the transformer was "
        "inferred from a tag on a pole.",
        "",
        "## Data-quality issues",
        "",
        f"The checks flagged **{qi['total_rows']:,}** issue rows"
        + (f" involving {qi['distinct_osm_objects_flagged']:,} distinct OSM objects (this counts every object "
           "involved, including the partner way of an overlap and every member way of a floating section, so it "
           "is not a count of wrong objects)" if qi.get("distinct_osm_objects_flagged") else "")
        + ". The row count is not an error count: most rows are 'to check' items (e.g. dangling line ends and "
        "disconnected sections, which may partly reflect incomplete OSM mapping) and tag gaps, "
        "not confirmed mistakes. None has been verified against the real network.",
        "",
        "By group:",
        "",
        _table(["Group", "Rows flagged"], [[GROUP_LABELS.get(k, k), f"{v:,}"] for k, v in qi["by_group"].items()]),
        "",
    ]
    le = qi["by_group"].get("likely_error", 0)
    if qi.get("likely_error_outside_substations") is not None:
        md += [
            f"Of the {le:,} rows in the 'likely error' group, **{qi['likely_error_inside_substations']:,} are inside "
            f"substation sites** (overlapping ways within a substation site) and **{qi['likely_error_outside_substations']:,} are elsewhere**. These are the ones "
            "flagged by the checks as probable mapping problems, not confirmed errors.",
            "",
        ]
    if qi.get("overlapping_lines_total") is not None:
        md += [f"Overlapping-lines check: {qi['overlapping_lines_total']:,} flagged, "
               f"{_fmt(qi['overlapping_lines_inside_substations'])} of them inside "
               f"{_fmt(qi['overlapping_lines_substation_sites'])} substation site(s) and the rest elsewhere.", ""]
    md += [
        "By check:",
        "",
        _table(["Check", "Rows flagged"], [[k.replace("_", " "), f"{v:,}"] for k, v in qi["by_check"].items()]),
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
    f = s.get("feeder")
    if f:
        b, d = f["base"], f.get("doubled")
        md += [
            "## Toy 11 kV feeder load flow (assumed loads and cable type; not a real utility feeder study)",
            "",
            "A pandapower toy model on one 11 kV path taken from the OSM data. Route and lengths come from "
            "OpenStreetMap; every electrical parameter is assumed (see the assumptions table in "
            "[outputs/feeder_results.md](feeder_results.md)). The results below describe the toy model only, "
            "not any real feeder.",
            "",
            f"- Substation: **{f.get('substation') or 'n/a'}** (OSM {_fmt(f.get('substation_osm_id'))})"
            + (" (fallback path: no zone substation touches an 11 kV-tagged line)" if f.get("fallback_path") else ""),
            f"- Path: **{_fmt(f['path_length_km'], '.2f', ' km')}** along {f['n_osm_ways']} OSM ways, "
            f"**{_fmt(f['n_buses'])} buses** ({_fmt(f['n_model_sections'])} model sections). Trunk only: branches "
            "are ignored."
            + (f" Only {f['n_zone_substations_with_11kv']} of {f['n_zone_substations']} zone-substation "
               "features touch any line tagged 11 kV, so the choice was narrow."
               if f.get("n_zone_substations_with_11kv") is not None and f.get("n_zone_substations") else ""),
            f"- Assumed cable: {f.get('cable_std_type')} on every section; {_fmt(f.get('n_loads'))} loads of "
            f"{_fmt(f.get('load_kw_each'), '.0f', ' kW')} each at power factor {_fmt(f.get('power_factor'))}, "
            f"ideal {_fmt(f.get('source_vm_pu'), '.2f')} pu source, no transformers.",
            "",
        ]
        rows = [["Total load (kW)", _fmt(b["total_load_kw"], ",.0f"), _fmt(d and d["total_load_kw"], ",.0f")],
                ["End-of-feeder voltage (pu)", _fmt(b["end_voltage_pu"], ".3f"), _fmt(d and d["end_voltage_pu"], ".3f")],
                ["End-of-feeder voltage drop (%)", _fmt(b["end_voltage_drop_pct"], ".2f"),
                 _fmt(d and d["end_voltage_drop_pct"], ".2f")],
                ["Max cable loading (% of rating)", _fmt(b["max_line_loading_pct"], ".1f"),
                 _fmt(d and d["max_line_loading_pct"], ".1f")],
                ["Losses (kW)", _fmt(b["total_losses_kw"], ".1f"), _fmt(d and d["total_losses_kw"], ".1f")]]
        md += [_table(["Result (toy model)", "Base loads", "Loads doubled"], rows), "",
               "Load multiplier (x the base loads) at which the toy model reaches a reference limit: "
               f"voltage below {_fmt(f.get('v_min_pu_reference'))} pu at **{_fmt(f.get('voltage_limit_multiplier'), '.2f')}x**; "
               f"cable loading above {_fmt(f.get('loading_limit_pct_reference'), '.0f')} % of rating at "
               f"**{_fmt(f.get('thermal_limit_multiplier'), '.2f')}x**. Binding limit: {f.get('binding_limit') or 'n/a'}. "
               "The limits are typical-practice references, not any network operator's standards.",
               ""]
    return "\n".join(md)


def duplicate_substation_notes(csv_path: Path) -> list[str]:
    """Details text of the duplicate_substation rows in quality_issues.csv (empty if unavailable)."""
    try:
        df = pd.read_csv(csv_path)
        rows = df[df["issue_type"] == "duplicate_substation"]
        return [f"{r['subtype'].replace('_', ' ')}: {r['details']}" for _, r in rows.iterrows()]
    except (OSError, ValueError, KeyError):
        return []


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
    feeder_results = load_feeder()
    summary = build_summary(lines, subs, len(areas), dists, quality, feeder_results,
                            duplicate_substation_notes(report.ISSUES_CSV))
    q_total = quality.get("network", {}).get("total_km_in_act")
    if q_total is not None and abs(q_total - summary["lines"]["total_km"]) > 0.15:
        print(f"WARNING: in-ACT km here ({summary['lines']['total_km']}) differs from quality_summary.json ({q_total})")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    SUMMARY_MD.write_text(render_markdown(summary) + "\n")
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
    _step("feeder (toy load flow)", feeder.main)
    _step("webmap", webmap.main)
    _step("static_map", static_map.main)
    print("\n=== summary ===")
    write_summary()
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
