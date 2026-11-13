"""Interactive folium map of the ACT power network -> ``outputs/map.html``.

Run with::

    python -m act_network.webmap

Reads the processed layers (EPSG:7855), simplifies geometry in metres, reprojects
to EPSG:4326 for Leaflet and writes a single self-contained HTML file (only the
JS/CSS that folium itself references is loaded).

Data provenance: public OpenStreetMap data (c) OpenStreetMap contributors (ODbL);
suburbs (c) ABS (CC BY 4.0). NOT Evoenergy network data; OSM power data is
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

from act_network import clean, service_areas

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MAP_HTML = OUTPUTS_DIR / "map.html"
ISSUES_LAYER = "quality_issues"  # topology.gpkg (see report.py)
TOPOLOGY_PATH = clean.PROCESSED_DIR / "topology.gpkg"

DISCLAIMER = (
    "Data: © OpenStreetMap contributors (ODbL); suburbs © ABS (CC BY 4.0). "
    "Public OSM data — NOT Evoenergy network data; OSM power data is incomplete. "
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
    "dangling_end": "#E31A1C",
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
    """GeoJSON dict with only ``keep`` properties and no feature ids."""
    return json.loads(gdf[keep + ["geometry"]].to_json(drop_id=True, show_bbox=False))


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
    for el, oid, power, v, band, op, name, circ, cab, loc, ug, ln in zip(
        _col(lines, "element"), _col(lines, "osm_id"), _col(lines, "power"), _col(lines, "voltage"),
        _col(lines, "voltage_band"), _col(lines, "operator"), _col(lines, "name"),
        _col(lines, "circuits"), _col(lines, "cables"), _col(lines, "location"),
        _col(lines, "is_underground"), _col(lines, "length_m"),
    ):
        rows = [("power", power), ("voltage tag", v if _has(v) else "missing"),
                ("voltage band", band), ("operator", op), ("name", name),
                ("circuits", circ), ("cables", cab), ("location", loc),
                ("underground", "yes" if ug is True or ug == 1 else None),
                ("length", f"{ln:,.0f} m" if _has(ln) else None)]
        out.append(_popup("Power line / cable", rows, osm_link(el, oid)))
    return out


def _line_layer(lines: gpd.GeoDataFrame, name: str, *, show: bool = True, dashed: bool = False):
    g = prepare(lines, LINE_TOL_M)
    g = g.assign(popup=line_popups(g), band=g["voltage_band"].fillna("unknown"))

    def style(f):
        band = f["properties"]["band"]
        s = {"color": BAND_COLOURS.get(band, "#7F7F7F"), "weight": BAND_WEIGHTS.get(band, 1.5), "opacity": 0.9}
        if dashed:
            s["dashArray"] = "6 4"
        return s

    return _popup_layer(_geojson(g, ["popup", "band"]), style, name=name, show=show, highlight=True)


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
                        marker=folium.CircleMarker(radius=5))


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
        marker = folium.CircleMarker(radius=7)
    else:
        def style(f):
            c = ISSUE_COLOURS.get(f["properties"]["itype"], "#666666")
            return {"color": c, "weight": 1, "fillColor": c, "fillOpacity": 0.6, "opacity": 0.8, "radius": 3}
        marker = folium.CircleMarker(radius=3)
    return _popup_layer(_geojson(g, ["popup", "itype"]), style, name=name, show=show, marker=marker,
                        highlight=use_geometry)


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


def legend_html() -> str:
    rows = [f'{_swatch(BAND_COLOURS[b])}{b}' for b in ("330 kV", "132 kV", "66 kV", "22 kV", "11 kV", "LV (415 V)", "unknown")]
    rows.append(f'{_swatch("#444", dashed=True)}Underground cable (dashed)')
    rows.append("<hr style='margin:4px 0'>")
    rows.append(f'{_swatch(SUB_STYLE["transmission"][0], "dot")}Transmission substation')
    rows.append(f'{_swatch(SUB_STYLE["zone"][0], "dot")}Zone substation')
    rows.append(f'{_swatch(SUB_STYLE["distribution"][0], "dot")}Distribution substation')
    rows.append(f'{_swatch("#8DD3C7", "box")}Rough nearest-substation area<br>'
                '<span style="margin-left:28px;font-size:10px">(geometric estimate, not a feeder boundary)</span>')
    rows.append("<hr style='margin:4px 0'>")
    rows.append(f'{_swatch("#E31A1C", "dot")}Likely error (on)')
    rows.append(f'{_swatch("#FF7F00", "dot")}To check: dangling end / floating / conflict (off)')
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
) -> folium.Map:
    """Assemble the map. All inputs are in a projected CRS (EPSG:7855)."""
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

    grp = issues["group"] if "group" in issues.columns else pd.Series(LIKELY_ERROR, index=issues.index)
    issue_layer(issues[grp == TAG_GAP], LAYER_NAMES["tag_gap"], show=False, use_geometry=False).add_to(m)
    issue_layer(issues[grp == TO_CHECK], LAYER_NAMES["check"], show=False, use_geometry=False).add_to(m)
    issue_layer(issues[grp == LIKELY_ERROR], LAYER_NAMES["likely"], show=True, use_geometry=True).add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    root = m.get_root().html
    root.add_child(folium.Element(disclaimer_html()))
    root.add_child(folium.Element(legend_html()))
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

    m = build_map(lines, subs, areas, suburbs, issues, boundary)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(MAP_HTML))
    print(f"Wrote {MAP_HTML} ({MAP_HTML.stat().st_size / 1e6:.1f} MB)")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
