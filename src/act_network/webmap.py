"""Interactive folium map of the ACT power network -> ``outputs/map.html``.

Run with::

    python -m act_network.webmap

Reads the processed layers (EPSG:7855), simplifies geometry in metres, reprojects
to EPSG:4326 for Leaflet and writes a single self-contained HTML file (only the
JS/CSS that folium itself references is loaded).

Data provenance: public OpenStreetMap data (c) OpenStreetMap contributors (ODbL);
suburbs (c) ABS (CC BY 4.0). NOT official network operator data; OSM power data is
incomplete; service areas are rough geometric estimates, not feeder boundaries.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

from act_network import clean, service_areas

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MAP_HTML = OUTPUTS_DIR / "map.html"
FEEDER_JSON = OUTPUTS_DIR / "feeder_results.json"
ISSUES_LAYER = "quality_issues"  # topology.gpkg (see report.py)
TOPOLOGY_PATH = clean.PROCESSED_DIR / "topology.gpkg"

DISCLAIMER = (
    "Data: © OpenStreetMap contributors (ODbL); suburbs © ABS (CC BY 4.0). "
    "Public OSM data — NOT official network operator data; OSM power data is incomplete. "
    "Service areas are rough geometric estimates, not feeder boundaries."
)

# Geometry simplification tolerances (metres, applied in EPSG:7855 before reprojecting)
LINE_TOL_M = 3.0
POLY_TOL_M = 25.0
COORD_PRECISION = 1e-5  # degrees, ~1 m

# Okabe-Ito based, colour-blind tolerant palette
BAND_COLOURS = {
    "330 kV": "#D55E00",
    "132 kV": "#E69F00",
    "66 kV": "#CC79A7",
    "22 kV": "#009E73",
    "11 kV": "#0072B2",
    "other": "#6A3D9A",
    "LV (415 V)": "#56B4E9",
    "unknown": "#7F7F7F",
}
BAND_WEIGHTS = {"330 kV": 3.5, "132 kV": 3.0, "66 kV": 2.5, "22 kV": 2.0, "11 kV": 2.0,
                "other": 2.0, "LV (415 V)": 1.0, "unknown": 1.5}
SUB_STYLE = {  # sub_class -> (colour, radius)
    "transmission": ("#B2182B", 9),
    "zone": ("#FFD92F", 7),
    "distribution": ("#1B1B1B", 3.5),
}
ISSUE_COLOURS = {
    "dangling_end": "#FF7F00",
    "floating_component": "#FF7F00",
    "voltage_conflict": "#984EA3",
    "overlapping_lines": "#E31A1C",
    "duplicate_substation": "#E31A1C",
    "missing_voltage": "#666666",
}
LIKELY_ERROR = "likely_error"
TO_CHECK = "to_check"
TAG_GAP = "tag_gap"

LAYER_NAMES = {
    "transmission": "Transmission (≥66 kV)",
    "distribution": "Distribution 11–22 kV",
    "lv": "LV 415 V",
    "unknown": "Unknown voltage",
    "underground": "Underground cables (dashed)",
    "substations": "Substations",
    "areas": "Rough nearest-substation areas",
    "suburbs": "Suburb boundaries",
    "boundary": "ACT boundary",
    "likely": "Data-quality: likely errors",
    "check": "Data-quality: to check",
    "tag_gap": "Data-quality: tag gaps (missing voltage)",
}
# Optional layer, only present when outputs/feeder_results.json exists (kept out of LAYER_NAMES)
FEEDER_LAYER_NAME = "Toy feeder path (load-flow model)"
FEEDER_COLOUR = "#E7298A"
INTERNAL_WEIGHT = 0.8  # substation bays / busbars are drawn thin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _has(value) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def osm_link(element, osm_id) -> str:
    """HTML link to the OSM object (way/relation/node)."""
    el = element if element in ("way", "relation", "node") else "way"
    try:
        oid = int(osm_id)
    except (TypeError, ValueError):
        return "n/a"
    return f'<a href="https://www.openstreetmap.org/{el}/{oid}" target="_blank" rel="noopener">{el} {oid}</a>'


def _popup(title: str, rows: list[tuple[str, object]], link: str | None = None) -> str:
    """Popup HTML. Every OSM-derived value is escaped (tags are untrusted free text)."""
    parts = [f"<b>{_esc(title)}</b>"]
    if link:
        parts.append(f"OSM: {link}")
    for label, value in rows:
        if _has(value):
            parts.append(f"{_esc(label)}: {_esc(value)}")
    return "<br>".join(parts)


def prepare(gdf: gpd.GeoDataFrame, tolerance: float = 0.0) -> gpd.GeoDataFrame:
    """Simplify in the projected CRS (metres), reproject to EPSG:4326, round coordinates."""
    if gdf.empty:
        return gdf.to_crs(4326) if gdf.crs else gdf
    g = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if tolerance > 0:
        g["geometry"] = g.geometry.simplify(tolerance, preserve_topology=True)
    g = g.to_crs(4326)
    g["geometry"] = shapely.set_precision(g.geometry.values, COORD_PRECISION)
    return g[~g.geometry.is_empty]


def _col(gdf: gpd.GeoDataFrame, name: str) -> list:
    return gdf[name].tolist() if name in gdf.columns else [None] * len(gdf)


def _geojson(gdf: gpd.GeoDataFrame, keep: list[str]) -> dict:
    """GeoJSON dict with only ``keep`` properties and short sequential feature ids.

    The ids matter for size: without a feature id folium keys its per-feature style
    ``switch`` on the first unique property (the long popup HTML), duplicating it.
    """
    data = json.loads(gdf[keep + ["geometry"]].to_json(drop_id=True, show_bbox=False))
    for i, feat in enumerate(data["features"]):
        feat["id"] = i
    return data


def _popup_layer(data: dict, style_function, *, name: str, show: bool = True, marker=None,
                 highlight: bool = False) -> folium.GeoJson:
    return folium.GeoJson(
        data,
        name=name,
        show=show,
        style_function=style_function,
        marker=marker,
        popup=folium.GeoJsonPopup(fields=["popup"], labels=False, max_width=360),
        highlight_function=(lambda f: {"weight": 5, "opacity": 1}) if highlight else None,
        smooth_factor=1.0,
    )


# ---------------------------------------------------------------------------
# layer builders (inputs in EPSG:7855)
# ---------------------------------------------------------------------------


def line_popups(lines: gpd.GeoDataFrame) -> list[str]:
    out = []
    for el, oid, power, v, band, op, name, circ, cab, loc, ug, ln, ln_act, tag, internal in zip(
        _col(lines, "element"), _col(lines, "osm_id"), _col(lines, "power"), _col(lines, "voltage"),
        _col(lines, "voltage_band"), _col(lines, "operator"), _col(lines, "name"),
        _col(lines, "circuits"), _col(lines, "cables"), _col(lines, "location"),
        _col(lines, "is_underground"), _col(lines, "length_m"), _col(lines, "length_in_act_m"),
        _col(lines, "line"), _col(lines, "is_substation_internal"),
    ):
        is_internal = internal is True or internal == 1
        if _has(ln_act) and _has(ln) and abs(float(ln) - float(ln_act)) > 1.0:
            length_rows = [("length inside ACT", f"{ln_act:,.0f} m"),
                           ("whole mapped way (incl. NSW section)", f"{ln:,.0f} m")]
        else:
            length_rows = [("length", f"{ln_act if _has(ln_act) else ln:,.0f} m" if _has(ln_act) or _has(ln) else None)]
        rows = [("power", power), ("line tag", tag),
                ("voltage tag", v if _has(v) else "missing"),
                ("voltage band", band), ("operator", op), ("name", name),
                ("circuits", circ), ("cables", cab), ("location", loc),
                ("underground", "yes" if ug is True or ug == 1 else None)] + length_rows
        title = "Substation bay / busbar (inside a substation site)" if is_internal else "Power line / cable"
        out.append(_popup(title, rows, osm_link(el, oid)))
    return out


def _line_layer(lines: gpd.GeoDataFrame, name: str, *, show: bool = True, dashed: bool = False):
    g = prepare(lines, LINE_TOL_M)
    internal = (g["is_substation_internal"].fillna(False).astype(bool)
                if "is_substation_internal" in g.columns else pd.Series(False, index=g.index))
    g = g.assign(popup=line_popups(g), band=g["voltage_band"].fillna("unknown"), internal=internal)

    def style(f):
        band = f["properties"]["band"]
        thin = f["properties"].get("internal")
        s = {"color": BAND_COLOURS.get(band, "#7F7F7F"),
             "weight": INTERNAL_WEIGHT if thin else BAND_WEIGHTS.get(band, 1.5),
             "opacity": 0.7 if thin else 0.9}
        if dashed:
            s["dashArray"] = "6 4"
        return s

    return _popup_layer(_geojson(g, ["popup", "band", "internal"]), style, name=name, show=show, highlight=True)


def substation_layer(subs: gpd.GeoDataFrame) -> folium.GeoJson:
    g = prepare(subs)
    pops = []
    for el, oid, nm, op, v, sc, gt in zip(
        _col(g, "element"), _col(g, "osm_id"), _col(g, "name"), _col(g, "operator"),
        _col(g, "voltage"), _col(g, "sub_class"), _col(g, "osm_geom_type"),
    ):
        pops.append(_popup(nm if _has(nm) else "Substation (unnamed)",
                           [("class", f"{sc} substation"), ("operator", op),
                            ("voltage tag", v if _has(v) else "missing"), ("mapped as", gt)],
                           osm_link(el, oid)))
    g = g.assign(popup=pops, sub_class=g["sub_class"].fillna("distribution") if "sub_class" in g else "distribution")

    def style(f):
        colour, radius = SUB_STYLE.get(f["properties"]["sub_class"], SUB_STYLE["distribution"])
        return {"color": "#222222", "weight": 1, "fillColor": colour, "fillOpacity": 0.95, "radius": radius}

    return _popup_layer(_geojson(g, ["popup", "sub_class"]), style, name=LAYER_NAMES["substations"],
                        marker=folium.CircleMarker(radius=5, fill=True))


def area_layer(areas: gpd.GeoDataFrame) -> folium.GeoJson:
    g = prepare(areas, POLY_TOL_M)
    pops = []
    for nm, oid, ids, a, lab, note in zip(
        _col(g, "name"), _col(g, "osm_id"), _col(g, "osm_ids"), _col(g, "area_km2"),
        _col(g, "label"), _col(g, "note"),
    ):
        pops.append(_popup(f"Rough nearest-substation area: {nm}",
                           [("area", f"{a:,.1f} km²" if _has(a) else None),
                            ("substation OSM id(s)", ids if _has(ids) else oid),
                            ("note", note if _has(note) else
                             "Geometric (Voronoi) estimate of the nearest zone substation. "
                             "NOT a feeder or supply boundary.")]))
    g = g.assign(popup=pops, name=g["name"] if "name" in g else "")
    palette = ["#8DD3C7", "#BEBADA", "#FDB462", "#B3DE69", "#FCCDE5", "#80B1D3", "#FFED6F", "#D9D9D9"]
    colours = {n: palette[i % len(palette)] for i, n in enumerate(sorted(set(g["name"])))}
    g = g.assign(fill=g["name"].map(colours))
    return _popup_layer(
        _geojson(g, ["popup", "fill"]),
        lambda f: {"color": "#555555", "weight": 1, "dashArray": "4 3", "fillColor": f["properties"]["fill"],
                   "fillOpacity": 0.15},
        name=LAYER_NAMES["areas"],
    )


def suburb_layer(suburbs: gpd.GeoDataFrame) -> folium.GeoJson:
    g = prepare(suburbs, POLY_TOL_M)
    pops = [_popup(nm if _has(nm) else "Suburb", [("ABS SAL code", code)])
            for nm, code in zip(_col(g, "suburb_name"), _col(g, "SAL_CODE21"))]
    g = g.assign(popup=pops)
    return _popup_layer(
        _geojson(g, ["popup"]),
        lambda f: {"color": "#444444", "weight": 1, "fillColor": "#000000", "fillOpacity": 0.0},
        name=LAYER_NAMES["suburbs"], show=False,
    )


def boundary_layer(boundary: gpd.GeoDataFrame) -> folium.GeoJson:
    g = prepare(boundary, POLY_TOL_M).assign(popup="ACT boundary (OpenStreetMap)")
    return _popup_layer(
        _geojson(g, ["popup"]),
        lambda f: {"color": "#000000", "weight": 2, "fillOpacity": 0.0},
        name=LAYER_NAMES["boundary"],
    )


def issue_popups(issues: pd.DataFrame) -> list[str]:
    out = []
    for t, grp, sub, el, oid, v, det, comp in zip(
        _col(issues, "issue_type"), _col(issues, "group"), _col(issues, "subtype"), _col(issues, "element"),
        _col(issues, "osm_id"), _col(issues, "voltage"), _col(issues, "details"), _col(issues, "component_id"),
    ):
        det = str(det)[:300] + ("…" if len(str(det)) > 300 else "") if _has(det) else None
        title = str(t).replace("_", " ").capitalize() if _has(t) else "Issue"
        out.append(_popup(title, [("group", str(grp).replace("_", " ")), ("subtype", sub),
                                  ("voltage tag", v), ("details", det)], osm_link(el, oid)))
    return out


def issues_to_points(issues: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace each issue's geometry with a representative point (lon/lat columns if present)."""
    g = issues.to_crs(4326).copy()
    if {"lon", "lat"} <= set(g.columns) and g[["lon", "lat"]].notna().all(axis=None):
        g["geometry"] = gpd.points_from_xy(g["lon"], g["lat"], crs=4326)
    else:
        g["geometry"] = g.geometry.representative_point()
    return g


def issue_layer(issues: gpd.GeoDataFrame, name: str, *, show: bool, use_geometry: bool) -> folium.GeoJson:
    if use_geometry:
        g = prepare(issues, LINE_TOL_M)
    else:
        g = issues_to_points(issues)
        g["geometry"] = shapely.set_precision(g.geometry.values, COORD_PRECISION)
    g = g.assign(popup=issue_popups(g), itype=g["issue_type"].fillna("")) if len(g) else g.assign(popup=[], itype=[])

    if use_geometry:
        def style(f):
            c = ISSUE_COLOURS.get(f["properties"]["itype"], "#E31A1C")
            return {"color": "#E31A1C", "weight": 5, "opacity": 0.9, "fillColor": c, "fillOpacity": 0.5, "radius": 7}
        marker = folium.CircleMarker(radius=7, fill=True)
    else:
        def style(f):
            c = ISSUE_COLOURS.get(f["properties"]["itype"], "#666666")
            return {"color": c, "weight": 1, "fillColor": c, "fillOpacity": 0.6, "opacity": 0.8, "radius": 3}
        marker = folium.CircleMarker(radius=3, fill=True)
    return _popup_layer(_geojson(g, ["popup", "itype"]), style, name=name, show=show, marker=marker,
                        highlight=use_geometry)


# ---------------------------------------------------------------------------
# toy feeder layer (from outputs/feeder_results.json)
# ---------------------------------------------------------------------------

TOY_NOTE = ("TOY MODEL: assumed loads and cable type (pandapower). Route and lengths come from public OSM data. "
            "Not a real utility feeder study.")


def load_feeder_results(path: Path | None = None) -> dict | None:
    """Parsed feeder_results.json, or None if missing / unreadable / without a path."""
    p = Path(path) if path is not None else FEEDER_JSON
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("path"), dict):
        return None
    return data if len(data["path"].get("bus_xy_epsg7855") or []) >= 2 else None


def _section_geometry(sec: dict, osm_sections: list, lines: gpd.GeoDataFrame | None,
                      a: tuple, b: tuple) -> LineString:
    """Section geometry in EPSG:7855: the OSM way geometry between the two buses if it can be
    recovered reliably, otherwise the straight chord between the buses."""
    chord = LineString([a, b])
    if lines is None or "geometry" not in lines.columns:
        return chord
    geoms = []
    for i in sec.get("osm_section_indices") or []:
        try:
            o = osm_sections[i]
            li, oid = int(o["line_idx"]), int(o["osm_id"])
            if 0 <= li < len(lines) and int(lines["osm_id"].iat[li]) == oid:
                geoms.append(lines.geometry.iat[li])
        except (IndexError, KeyError, TypeError, ValueError):
            return chord
    if not geoms:
        return chord
    merged = linemerge(list(dict.fromkeys(geoms))) if len(geoms) > 1 else geoms[0]
    if merged.geom_type != "LineString":
        return chord
    d0, d1 = merged.project(Point(a)), merged.project(Point(b))
    if d0 == d1:
        return chord
    part = substring(merged, d0, d1)
    want = float(sec.get("length_m") or 0)
    if part.geom_type != "LineString" or (want and abs(part.length - want) > max(25.0, 0.1 * want)):
        return chord
    return part


def feeder_layer(feeder: dict, lines: gpd.GeoDataFrame | None = None, *, show: bool = True) -> folium.GeoJson | None:
    """Toy feeder path: one line per model section plus a marker per bus. None if unusable.

    ``lines`` (EPSG:7855, same row order as when the feeder was built) lets the sections follow the
    OSM way geometry; without it the sections are straight lines between buses.
    """
    path = feeder.get("path", {})
    bus_xy = path.get("bus_xy_epsg7855") or []
    secs = path.get("model_sections") or []
    if len(bus_xy) < 2 or str(path.get("crs", "EPSG:7855")).upper() != "EPSG:7855":
        return None
    res = feeder.get("results", {})
    base, dbl = res.get("base", {}), res.get("doubled", {})
    model = feeder.get("model", {})

    def pu(r, key, i):
        try:
            return f"{r[key][i]:.3f} pu"
        except (KeyError, IndexError, TypeError):
            return None

    def pct(r, i):
        try:
            return f"{r['line_loading_pct'][i]:.0f} %"
        except (KeyError, IndexError, TypeError):
            return None

    geoms, props = [], []
    sub = path.get("substation_name")
    load_buses = set(model.get("load_buses") or [])
    for i, (x, y) in enumerate(bus_xy):
        role = f"Source bus: {sub}" if i == 0 and sub else ("Source bus" if i == 0 else f"Bus {i}")
        rows = [("assumed load", f"{model['load_kw']:g} kW at pf {model['power_factor']:g}"
                 if i in load_buses and "load_kw" in model and "power_factor" in model else None),
                ("voltage, base loads", pu(base, "bus_vm_pu", i)), ("voltage, loads doubled", pu(dbl, "bus_vm_pu", i)),
                ("distance from source", f"{base['bus_distance_km'][i]:.2f} km"
                 if i < len(base.get("bus_distance_km") or []) else None)]
        geoms.append(Point(x, y))
        props.append({"popup": _popup(f"Toy feeder: {role}", [("note", TOY_NOTE)] + rows), "kind": "bus"})
    osm_sections = path.get("osm_sections") or []
    for k, sec in enumerate(secs):
        if k + 1 >= len(bus_xy):
            break
        geom = _section_geometry(sec, osm_sections, lines, tuple(bus_xy[k]), tuple(bus_xy[k + 1]))
        ways = ", ".join(osm_link("way", w) for w in sec.get("osm_ids") or [])
        rows = [("note", TOY_NOTE), ("length", f"{sec['length_m']:,.0f} m" if _has(sec.get("length_m")) else None),
                ("assumed cable", model.get("std_type")),
                ("loading, base loads", pct(base, k)), ("loading, loads doubled", pct(dbl, k))]
        popup = _popup(f"Toy feeder: section {sec.get('section', k + 1)}", rows)
        if ways:
            popup += "<br>OSM ways: " + ways  # links built by osm_link (ids are ints)
        geoms.append(geom)
        props.append({"popup": popup, "kind": "section"})
    g = gpd.GeoDataFrame(props, geometry=geoms, crs=7855)
    g = prepare(g)

    def style(f):
        if f["properties"]["kind"] == "bus":
            return {"color": "#222222", "weight": 1, "fillColor": FEEDER_COLOUR, "fillOpacity": 1, "radius": 5}
        return {"color": FEEDER_COLOUR, "weight": 6, "opacity": 0.85}

    return _popup_layer(_geojson(g, ["popup", "kind"]), style, name=FEEDER_LAYER_NAME, show=show,
                        marker=folium.CircleMarker(radius=5, fill=True), highlight=True)


# ---------------------------------------------------------------------------
# legend / disclaimer
# ---------------------------------------------------------------------------


def _swatch(colour: str, kind: str = "line", dashed: bool = False) -> str:
    if kind == "dot":
        return f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{colour};border:1px solid #222;margin-right:6px;vertical-align:middle"></span>'
    if kind == "box":
        return f'<span style="display:inline-block;width:18px;height:11px;background:{colour};opacity:.5;border:1px dashed #555;margin-right:6px;vertical-align:middle"></span>'
    style = "dashed" if dashed else "solid"
    return f'<span style="display:inline-block;width:22px;border-top:3px {style} {colour};margin-right:6px;vertical-align:middle"></span>'


def legend_html(feeder: bool = False) -> str:
    rows = [f'{_swatch(BAND_COLOURS[b])}{b}' for b in ("330 kV", "132 kV", "66 kV", "22 kV", "11 kV", "LV (415 V)", "unknown")]
    rows.append(f'{_swatch("#444", dashed=True)}Underground cable (dashed)')
    rows.append("<hr style='margin:4px 0'>")
    rows.append(f'{_swatch(SUB_STYLE["transmission"][0], "dot")}Transmission substation')
    rows.append(f'{_swatch(SUB_STYLE["zone"][0], "dot")}Zone substation')
    rows.append(f'{_swatch(SUB_STYLE["distribution"][0], "dot")}Distribution substation')
    rows.append(f'{_swatch("#8DD3C7", "box")}Rough nearest-substation area<br>'
                '<span style="margin-left:28px;font-size:10px">(geometric estimate, not a feeder boundary)</span>')
    if feeder:
        rows.append(f'{_swatch(FEEDER_COLOUR)}Toy feeder path (assumed-load model, not a real feeder study)')
    rows.append(f'{_swatch("#444")}Thin line: substation bay / busbar'.replace("border-top:3px", "border-top:1px"))
    rows.append("<hr style='margin:4px 0'>")
    rows.append(f'{_swatch("#E31A1C", "dot")}Likely error (on)')
    rows.append(f'{_swatch("#FF7F00", "dot")}To check: dangling end / floating section (off)')
    rows.append(f'{_swatch(ISSUE_COLOURS["voltage_conflict"], "dot")}To check: voltage conflict (off)')
    rows.append(f'{_swatch("#666666", "dot")}Tag gap: missing voltage (off)')
    body = "<br>".join(rows)
    return (
        '<details open id="act-legend" style="position:fixed;bottom:24px;left:10px;z-index:9999;'
        'background:rgba(255,255,255,.93);padding:6px 10px;border:1px solid #888;border-radius:4px;'
        'font:12px/1.5 Arial,sans-serif;color:#111;max-width:300px;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
        '<summary style="cursor:pointer;font-weight:bold">Legend</summary>' + body + "</details>"
    )


def disclaimer_html() -> str:
    return (
        '<div id="act-disclaimer" style="position:fixed;top:8px;left:50px;right:260px;z-index:9999;'
        'background:rgba(255,255,230,.95);padding:4px 10px;border:1px solid #888;border-radius:4px;'
        'font:11px/1.4 Arial,sans-serif;color:#111;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
        + _esc(DISCLAIMER) + "</div>"
    )


# ---------------------------------------------------------------------------
# map assembly
# ---------------------------------------------------------------------------


def build_map(
    lines: gpd.GeoDataFrame,
    substations: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    suburbs: gpd.GeoDataFrame,
    issues: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame | None = None,
    feeder: dict | None = None,
) -> folium.Map:
    """Assemble the map. All inputs are in a projected CRS (EPSG:7855).

    ``feeder`` is the parsed outputs/feeder_results.json; when None (or unusable) the toy-feeder
    layer is skipped.
    """
    ref = boundary if boundary is not None and len(boundary) else lines
    minx, miny, maxx, maxy = prepare(ref).total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9, tiles=None,
                   control_scale=True, prefer_canvas=True)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.TileLayer("OpenStreetMap", name="Base map: OpenStreetMap").add_to(m)

    suburb_layer(suburbs).add_to(m)
    area_layer(areas).add_to(m)
    if boundary is not None and len(boundary):
        boundary_layer(boundary).add_to(m)

    ug = lines["is_underground"].fillna(False).astype(bool)
    over = lines[~ug]
    band = over["voltage_band"].fillna("unknown")
    cls = over["line_class"].fillna("unknown")
    lv = band == clean.LV_BAND
    _line_layer(over[(cls == "distribution") & lv], LAYER_NAMES["lv"]).add_to(m)
    _line_layer(over[cls == "unknown"], LAYER_NAMES["unknown"]).add_to(m)
    _line_layer(over[(cls == "distribution") & ~lv], LAYER_NAMES["distribution"]).add_to(m)
    _line_layer(lines[ug], LAYER_NAMES["underground"], dashed=True).add_to(m)
    _line_layer(over[cls == "transmission"], LAYER_NAMES["transmission"]).add_to(m)
    substation_layer(substations).add_to(m)
    feeder_lyr = feeder_layer(feeder, lines) if feeder else None
    if feeder_lyr is not None:
        feeder_lyr.add_to(m)

    grp = issues["group"] if "group" in issues.columns else pd.Series(LIKELY_ERROR, index=issues.index)
    issue_layer(issues[grp == TAG_GAP], LAYER_NAMES["tag_gap"], show=False, use_geometry=False).add_to(m)
    issue_layer(issues[grp == TO_CHECK], LAYER_NAMES["check"], show=False, use_geometry=False).add_to(m)
    issue_layer(issues[grp == LIKELY_ERROR], LAYER_NAMES["likely"], show=True, use_geometry=True).add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    root = m.get_root().html
    root.add_child(folium.Element(disclaimer_html()))
    root.add_child(folium.Element(legend_html(feeder_lyr is not None)))
    m.get_root().header.add_child(folium.Element("<title>ACT electricity network (OSM) - explorer</title>"))
    return m


def main() -> int:
    p = clean.PROCESSED_PATH
    lines = gpd.read_file(p, layer=clean.LINES_LAYER)
    subs = gpd.read_file(p, layer=clean.SUBSTATIONS_LAYER)
    suburbs = gpd.read_file(p, layer=clean.SUBURBS_LAYER)
    boundary = gpd.read_file(p, layer=clean.BOUNDARY_LAYER)
    areas = gpd.read_file(service_areas.SERVICE_AREAS_PATH, layer=service_areas.AREAS_LAYER)
    issues = gpd.read_file(TOPOLOGY_PATH, layer=ISSUES_LAYER)

    feeder = load_feeder_results()
    if feeder is None:
        print(f"No usable {FEEDER_JSON.name}: toy feeder layer skipped (run `python -m act_network.feeder`).")
    m = build_map(lines, subs, areas, suburbs, issues, boundary, feeder)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(MAP_HTML))
    print(f"Wrote {MAP_HTML} ({MAP_HTML.stat().st_size / 1e6:.1f} MB)")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
