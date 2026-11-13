"""Offline tests for webmap.py and the summary maths in build.py (synthetic data only)."""

import datetime as dt

import folium
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from act_network import build, webmap

CRS = 7855
X0, Y0 = 690000.0, 6090000.0


def _lines():
    rows = [
        # osm_id, power, voltage, band, class, underground, length handled below
        (1, "line", "330000", "330 kV", "transmission", False, LineString([(X0, Y0), (X0 + 1000, Y0)])),
        (2, "minor_line", "11000", "11 kV", "distribution", False, LineString([(X0, Y0 + 10), (X0 + 500, Y0 + 10)])),
        (3, "minor_line", "415", "LV (415 V)", "distribution", False, LineString([(X0, Y0 + 20), (X0 + 250, Y0 + 20)])),
        (4, "minor_line", None, "unknown", "unknown", False, LineString([(X0, Y0 + 30), (X0 + 100, Y0 + 30)])),
        (5, "cable", "11000", "11 kV", "distribution", True, LineString([(X0, Y0 + 40), (X0 + 50, Y0 + 40)])),
    ]
    g = gpd.GeoDataFrame(
        {
            "osm_id": [r[0] for r in rows], "element": "way", "power": [r[1] for r in rows],
            "voltage": [r[2] for r in rows], "voltage_band": [r[3] for r in rows],
            "line_class": [r[4] for r in rows], "is_underground": [r[5] for r in rows],
            "name": ["<script>alert(1)</script>", None, None, None, None],
            "operator": "Evoenergy", "circuits": None, "cables": None, "location": None,
        },
        geometry=[r[6] for r in rows], crs=CRS,
    )
    g["length_m"] = g.geometry.length
    return g


def _subs():
    return gpd.GeoDataFrame(
        {"osm_id": [10, 11, 12], "element": "way", "name": ["TS", "Zone A", None], "operator": "Evoenergy",
         "voltage": ["330000", "132000;11000", None], "sub_class": ["transmission", "zone", "distribution"],
         "osm_geom_type": "Point"},
        geometry=[Point(X0, Y0), Point(X0 + 100, Y0 + 100), Point(X0 + 200, Y0 + 200)], crs=CRS)


def _poly(dx=0):
    return Polygon([(X0 + dx, Y0 - 500), (X0 + dx + 800, Y0 - 500), (X0 + dx + 800, Y0 + 500), (X0 + dx, Y0 + 500)])


def _areas():
    return gpd.GeoDataFrame({"osm_id": [11], "osm_ids": ["11"], "name": ["Zone A"], "area_km2": [1.0],
                             "label": ["rough nearest-substation area"], "note": ["rough"]},
                            geometry=[_poly()], crs=CRS)


def _suburbs():
    return gpd.GeoDataFrame({"SAL_CODE21": ["80001"], "suburb_name": ["Testville"], "is_remainder": [False]},
                            geometry=[_poly(100)], crs=CRS)


def _boundary():
    return gpd.GeoDataFrame({"name": ["ACT"]}, geometry=[_poly(-100).buffer(2000)], crs=CRS)


def _issues():
    pts = [Point(X0 + 5, Y0 + 5), Point(X0 + 6, Y0 + 6), Point(X0 + 7, Y0 + 7), LineString([(X0, Y0), (X0 + 30, Y0)])]
    g = gpd.GeoDataFrame(
        {"issue_type": ["overlapping_lines", "dangling_end", "missing_voltage", "dangling_end"],
         "group": ["likely_error", "to_check", "tag_gap", "to_check"],
         "subtype": ["same_voltage", "isolated", "", ""], "osm_id": [1, 2, 4, 3], "element": "way",
         "voltage": ["330000", "11000", None, "11000"], "details": ["overlaps way 2", "dangling", "no voltage", "x" * 500],
         "component_id": [1.0, 2.0, 3.0, 4.0]},
        geometry=pts, crs=CRS)
    g["lon"] = [149.1, 149.11, 149.12, 149.13]
    g["lat"] = [-35.3, -35.31, -35.32, -35.33]
    return g


@pytest.fixture(scope="module")
def fmap():
    return webmap.build_map(_lines(), _subs(), _areas(), _suburbs(), _issues(), _boundary())


@pytest.fixture(scope="module")
def html_text(fmap):
    return fmap.get_root().render()


def _overlays(fmap):
    return {c.layer_name: c for c in fmap._children.values() if isinstance(c, folium.GeoJson)}


def test_expected_layer_names(fmap):
    assert set(_overlays(fmap)) == set(webmap.LAYER_NAMES.values())


def test_every_layer_is_in_layer_control(fmap, html_text):
    # Each overlay is registered with the LayerControl, so each one toggles
    start = html_text.index("overlays :")
    block = html_text[start:html_text.index("}", start)]
    assert block.count("geo_json_") == len(webmap.LAYER_NAMES)
    assert any(isinstance(c, folium.LayerControl) for c in fmap._children.values())


def test_default_visibility(fmap):
    layers = _overlays(fmap)
    off = {n for n, c in layers.items() if not c.show}
    assert off == {webmap.LAYER_NAMES[k] for k in ("suburbs", "check", "tag_gap")}
    assert layers[webmap.LAYER_NAMES["likely"]].show


def test_disclaimer_and_legend_present(html_text):
    for fragment in ("© OpenStreetMap contributors (ODbL)", "© ABS (CC BY 4.0)", "NOT Evoenergy network data",
                     "OSM power data is incomplete", "rough geometric estimates, not feeder boundaries"):
        assert fragment in html_text
    assert "act-legend" in html_text and "act-disclaimer" in html_text
    assert "Rough nearest-substation area" in html_text


def test_popups_link_to_osm_and_escape_tags(html_text):
    assert "https://www.openstreetmap.org/way/1" in html_text
    assert "<script>alert(1)</script>" not in html_text  # OSM name tag must be escaped


def test_osm_link_and_popup_helpers():
    assert 'href="https://www.openstreetmap.org/relation/5"' in webmap.osm_link("relation", 5)
    assert webmap.osm_link("way", None) == "n/a"
    popup = webmap._popup("T", [("a", "<b>"), ("empty", None)], "L")
    assert "&lt;b&gt;" in popup and "empty" not in popup


def test_prepare_reprojects_and_rounds():
    g = webmap.prepare(_lines(), tolerance=3.0)
    assert g.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = g.total_bounds
    assert 148 < minx < 150 and -36 < miny < -35
    for x, y in list(g.geometry.iloc[0].coords):
        assert round(x, 5) == pytest.approx(x, abs=1e-9)


def test_issue_points_use_lon_lat():
    pts = webmap.issues_to_points(_issues())
    assert set(pts.geom_type) == {"Point"}
    assert pts.geometry.iloc[0].x == pytest.approx(149.1)


# ---------------------------------------------------------------------------
# summary maths (build.py)
# ---------------------------------------------------------------------------


def test_summarise_lines_km():
    s = build.summarise_lines(_lines())
    assert s["n_line_segments"] == 5
    assert s["total_km"] == pytest.approx(1.9, abs=0.05)  # 1000+500+250+100+50 m
    assert s["km_by_voltage_band"]["330 kV"] == 1.0
    assert s["km_by_voltage_band"]["11 kV"] == 0.6  # 500 + 50 (underground cable counts)
    assert s["km_by_voltage_band"]["LV (415 V)"] == pytest.approx(0.25, abs=0.06)  # 250 m
    assert list(s["km_by_voltage_band"])[0] == "330 kV"
    assert s["km_by_line_class"] == {"transmission": 1.0, "distribution": 0.8, "unknown": 0.1}
    assert s["km_underground"] == 0.1 and s["n_underground_segments"] == 1


def test_summarise_substations():
    s = build.summarise_substations(_subs())
    assert s == {"total": 3, "by_sub_class": {"transmission": 1, "zone": 1, "distribution": 1}}


def test_top_furthest_excludes_remainders_and_sorts():
    d = pd.DataFrame({
        "suburb_name": ["A", "B", "ACT Remainder - X", "C", "D", "E", "F", "G"],
        "is_remainder": [False, False, True, False, False, False, False, False],
        "distance_km": [1.0, 9.0, 99.0, 5.0, 3.0, 7.0, 2.0, 4.0],
        "nearest_substation": "S",
    })
    top = build.top_furthest_suburbs(d)
    assert [r["suburb"] for r in top] == ["B", "E", "C", "G", "D"]
    assert top[0]["rank"] == 1 and top[0]["distance_km"] == 9.0


QUALITY = {"issues": {
    "total_rows": 10, "by_group": {"likely_error": 2, "to_check": 7, "tag_gap": 1},
    "by_type": {"dangling_end": 7, "overlapping_lines": 2, "missing_voltage": 1},
    "by_type_and_group": {"dangling_end/to_check": 7, "overlapping_lines/likely_error": 2,
                          "missing_voltage/tag_gap": 1},
    "distinct_osm_objects_flagged": 9}}


def test_build_summary_and_markdown(tmp_path):
    d = pd.DataFrame({"suburb_name": ["A"], "is_remainder": [False], "distance_km": [2.34],
                      "nearest_substation": "Zone A"})
    s = build.build_summary(_lines(), _subs(), 16, d, QUALITY, today=dt.date(2026, 1, 2))
    assert s["generated"] == "2026-01-02" and s["zone_service_areas"] == 16
    assert s["quality_issues"]["by_group"]["likely_error"] == 2
    md = build.render_markdown(s)
    assert "2026-01-02" in md and "NOT Evoenergy network data" in md
    assert "Toy 11 kV feeder" not in md  # hook omitted without feeder results
    md2 = build.render_markdown(s, {"end_voltage_pu": 0.97, "nested": {"x": 1}})
    assert "Toy 11 kV feeder" in md2 and "end_voltage_pu: 0.97" in md2 and "nested / x: 1" in md2


def test_load_feeder_absent_or_bad(tmp_path):
    assert build.load_feeder(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert build.load_feeder(bad) is None


def test_missing_raw_exits_with_message(monkeypatch, capsys):
    monkeypatch.setattr(build, "missing_raw_files", lambda: [build.OUTPUTS_DIR / "nope.gpkg"])
    assert build.main() == 1
    assert "python -m act_network.fetch" in capsys.readouterr().out
