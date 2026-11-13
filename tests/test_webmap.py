"""Offline tests for webmap.py and the summary maths in build.py (synthetic data only)."""

import datetime as dt
import json

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
        # osm_id, power, voltage, band, class, underground, geometry, length_in_act_m, line tag, bay/busbar
        (1, "line", "330000", "330 kV", "transmission", False, LineString([(X0, Y0), (X0 + 1000, Y0)]), 650.0, None, False),
        (2, "minor_line", "11000", "11 kV", "distribution", False, LineString([(X0, Y0 + 10), (X0 + 500, Y0 + 10)]), None, None, False),
        (3, "minor_line", "415", "LV (415 V)", "distribution", False, LineString([(X0, Y0 + 20), (X0 + 250, Y0 + 20)]), None, None, False),
        (4, "minor_line", None, "unknown", "unknown", False, LineString([(X0, Y0 + 30), (X0 + 100, Y0 + 30)]), None, None, False),
        (5, "cable", "11000", "11 kV", "distribution", True, LineString([(X0, Y0 + 40), (X0 + 50, Y0 + 40)]), None, None, False),
        (6, "line", "330000", "330 kV", "transmission", False, LineString([(X0, Y0 + 50), (X0 + 60, Y0 + 50)]), None, "bay", True),
    ]
    g = gpd.GeoDataFrame(
        {
            "osm_id": [r[0] for r in rows], "element": "way", "power": [r[1] for r in rows],
            "voltage": [r[2] for r in rows], "voltage_band": [r[3] for r in rows],
            "line_class": [r[4] for r in rows], "is_underground": [r[5] for r in rows],
            "line": [r[8] for r in rows], "is_substation_internal": [r[9] for r in rows],
            "name": ["<script>alert(1)</script>", None, None, None, None, None],
            "operator": "Evoenergy", "circuits": None, "cables": None, "location": None,
        },
        geometry=[r[6] for r in rows], crs=CRS,
    )
    g["length_m"] = g.geometry.length
    # only line 1 crosses the ACT boundary (650 m of its 1000 m are inside); the rest are fully inside
    g["length_in_act_m"] = [r[7] if r[7] is not None else g["length_m"].iat[i] for i, r in enumerate(rows)]
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


def test_point_markers_are_filled_and_styles_keyed_on_feature_id(fmap, html_text):
    # folium's CircleMarker defaults to fill=false, which draws substations / issue points hollow
    for name in (webmap.LAYER_NAMES["substations"], webmap.LAYER_NAMES["check"]):
        assert _overlays(fmap)[name].marker.options["fill"] is True
    assert '"fill": false' not in html_text
    # short feature ids, so folium's style switch is not keyed on (and does not duplicate) the popup HTML
    assert "switch(feature.properties.popup)" not in html_text
    assert all("id" in f for f in _overlays(fmap)[webmap.LAYER_NAMES["substations"]].data["features"])


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
# popups for the new line columns
# ---------------------------------------------------------------------------


def test_line_popups_show_line_tag_and_both_lengths():
    pops = webmap.line_popups(_lines())
    assert "length inside ACT: 650 m" in pops[0] and "whole mapped way (incl. NSW section): 1,000 m" in pops[0]
    assert "length: 500 m" in pops[1]  # fully inside: one length only
    assert "Substation bay / busbar" in pops[5] and "line tag: bay" in pops[5]


def test_bays_are_styled_thin():
    layer = webmap._line_layer(_lines(), "t")
    props = [f["properties"] for f in layer.data["features"]]
    assert [p["internal"] for p in props] == [False] * 5 + [True]
    thin = layer.style_function({"properties": props[5]})
    normal = layer.style_function({"properties": props[0]})
    assert thin["weight"] == webmap.INTERNAL_WEIGHT < min(webmap.BAND_WEIGHTS.values())
    assert normal["weight"] == webmap.BAND_WEIGHTS["330 kV"]


# ---------------------------------------------------------------------------
# toy feeder layer
# ---------------------------------------------------------------------------


def _feeder_lines():
    """Two ways that exist in the lines layer with the osm_ids/line_idx used in FEEDER (a curved first way)."""
    return gpd.GeoDataFrame(
        {"osm_id": [100, 101]},
        geometry=[LineString([(X0, Y0), (X0 + 500, Y0 + 200), (X0 + 1000, Y0)]),
                  LineString([(X0 + 1000, Y0), (X0 + 1500, Y0)])], crs=CRS)


def test_feeder_layer_follows_osm_way_when_lines_given():
    layer = webmap.feeder_layer(FEEDER, _feeder_lines())
    sec = [f for f in layer.data["features"] if f["properties"]["kind"] == "section"]
    assert len(sec) == 2 and len([f for f in layer.data["features"] if f["properties"]["kind"] == "bus"]) == 3
    assert len(sec[0]["geometry"]["coordinates"]) == 3  # curved way kept, not a straight chord
    assert len(sec[1]["geometry"]["coordinates"]) == 2


def test_feeder_layer_falls_back_to_chords_and_bad_geometry():
    layer = webmap.feeder_layer(FEEDER, None)
    sec = [f for f in layer.data["features"] if f["properties"]["kind"] == "section"]
    assert all(len(f["geometry"]["coordinates"]) == 2 for f in sec)
    # a way whose osm_id does not match the feeder file is ignored (chord used)
    bad = _feeder_lines().assign(osm_id=[7, 8])
    layer = webmap.feeder_layer(FEEDER, bad)
    assert all(len(f["geometry"]["coordinates"]) == 2 for f in layer.data["features"] if f["properties"]["kind"] == "section")
    bad_crs = json.loads(json.dumps(FEEDER))
    bad_crs["path"]["crs"] = "EPSG:4326"
    assert webmap.feeder_layer(bad_crs) is None


def test_map_with_feeder_has_toggle_layer_and_toy_popup(tmp_path):
    f = tmp_path / "feeder_results.json"
    f.write_text(json.dumps(FEEDER))
    feeder = webmap.load_feeder_results(f)
    assert feeder is not None
    m = webmap.build_map(_lines(), _subs(), _areas(), _suburbs(), _issues(), _boundary(), feeder)
    layers = _overlays(m)
    assert set(layers) == set(webmap.LAYER_NAMES.values()) | {webmap.FEEDER_LAYER_NAME}
    html_text = m.get_root().render()
    assert "TOY MODEL: assumed loads and cable type" in html_text
    assert "Toy feeder path (assumed-load model" in html_text  # legend entry
    start = html_text.index("overlays :")
    block = html_text[start:html_text.index("}", start)]
    assert block.count("geo_json_") == len(webmap.LAYER_NAMES) + 1  # registered in the LayerControl


def test_feeder_missing_or_unusable_is_skipped(tmp_path, fmap):
    assert webmap.load_feeder_results(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert webmap.load_feeder_results(bad) is None
    nopath = tmp_path / "nopath.json"
    nopath.write_text(json.dumps({"title": "x"}))
    assert webmap.load_feeder_results(nopath) is None
    assert webmap.FEEDER_LAYER_NAME not in _overlays(fmap)  # default build has no feeder layer


# ---------------------------------------------------------------------------
# summary maths (build.py)
# ---------------------------------------------------------------------------


def test_summarise_lines_headline_is_km_inside_act():
    s = build.summarise_lines(_lines())
    assert s["n_line_segments"] == 6
    # inside the ACT: 650 + 500 + 250 + 100 + 50 + 60 = 1610 m; whole ways: 1000 + ... = 1960 m
    assert s["total_km"] == pytest.approx(1.61, abs=0.05)
    assert s["total_km_whole_ways"] == pytest.approx(1.96, abs=0.05)
    assert s["total_km"] < s["total_km_whole_ways"]
    assert s["n_segments_crossing_act_boundary"] == 1 and s["n_segments_wholly_outside_act"] == 0
    assert s["km_by_voltage_band"]["330 kV"] == pytest.approx(0.71, abs=0.05)  # 650 + 60 bay
    assert s["km_by_voltage_band_whole_ways"]["330 kV"] == pytest.approx(1.06, abs=0.05)
    assert s["km_by_voltage_band"]["11 kV"] == pytest.approx(0.55, abs=0.051)  # underground cable counts
    assert s["km_by_voltage_band"]["LV (415 V)"] == pytest.approx(0.25, abs=0.06)
    assert list(s["km_by_voltage_band"])[0] == "330 kV"
    assert s["km_by_line_class"]["transmission"] == pytest.approx(0.71, abs=0.05)
    assert s["km_by_line_class_whole_ways"]["transmission"] == pytest.approx(1.06, abs=0.05)
    assert s["km_by_line_class"]["unknown"] == pytest.approx(0.1, abs=0.05)
    # both views sum to their own total (bands include every row)
    assert sum(s["km_by_voltage_band"].values()) == pytest.approx(s["total_km"], abs=0.1)
    assert sum(s["km_by_voltage_band_whole_ways"].values()) == pytest.approx(s["total_km_whole_ways"], abs=0.1)
    assert s["km_underground"] == pytest.approx(0.05, abs=0.051) and s["n_underground_segments"] == 1


def test_summarise_lines_bays_busbars_and_missing_column():
    s = build.summarise_lines(_lines())
    assert s["n_substation_bay_busbar_ways"] == 1
    assert s["km_substation_bays_busbars"] == pytest.approx(0.06, abs=0.001)
    assert s["km_substation_bays_busbars_by_line_class"] == {"transmission": pytest.approx(0.06, abs=0.001)}
    assert s["km_transmission_excl_bays_busbars"] == pytest.approx(0.65, abs=0.05)
    with pytest.raises(ValueError, match="length_in_act_m"):
        build.summarise_lines(_lines().drop(columns="length_in_act_m"))


def test_summarise_substations():
    s = build.summarise_substations(_subs())
    assert s["total"] == 3
    assert s["by_sub_class"] == {"transmission": 1, "zone": 1, "distribution": 1}
    assert s["zone_sites_by_name"] == 1 and s["n_unnamed"] == 1 and s["zone_duplicate_names"] == []


def test_zone_features_vs_sites_after_name_dedupe():
    subs = _subs()
    dup = subs.iloc[[1]].assign(osm_id=99, name="  ZONE a ")  # same zone site mapped twice
    s = build.summarise_substations(pd.concat([subs, dup], ignore_index=True), n_duplicate_features=1)
    assert s["by_sub_class"]["zone"] == 2 and s["zone_sites_by_name"] == 1
    assert s["zone_duplicate_names"] == ["ZONE a", "Zone A"]
    assert s["n_duplicate_features_flagged"] == 1


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


QUALITY = {
    "network": {"n_transformers": 5, "n_transformers_by_source": {"power=transformer": 3, "pole transformer tag": 2}},
    "checks": {"overlapping_lines": {"count": 2, "inside_substations": 1, "substation_sites": 1},
               "duplicate_substations": {"nested_footprint": 1, "same_name": 1}},
    "issues": {
        "total_rows": 10, "by_group": {"likely_error": 2, "to_check": 7, "tag_gap": 1},
        "likely_error_outside_substations": 1,
        "by_type": {"dangling_end": 7, "overlapping_lines": 2, "missing_voltage": 1},
        "by_type_and_group": {"dangling_end/to_check": 7, "overlapping_lines/likely_error": 2,
                              "missing_voltage/tag_gap": 1},
        "distinct_osm_objects_flagged": 9}}

FEEDER = {
    "title": "Toy model",
    "path": {
        "substation_name": "Zone A", "substation_osm_id": 11, "fallback": False, "total_length_m": 1500.0,
        "reachable_11kv_length_m": 2500.0, "n_zone_substations": 17, "n_zone_substations_with_11kv": 2,
        "n_osm_ways": 2, "n_buses": 3, "n_model_sections": 2, "osm_way_ids": [100, 101],
        "osm_sections": [{"osm_id": 100, "line_idx": 0, "length_m": 1000.0},
                         {"osm_id": 101, "line_idx": 1, "length_m": 500.0}],
        "model_sections": [{"section": 1, "length_m": 1000.0, "osm_ids": [100], "osm_section_indices": [0]},
                           {"section": 2, "length_m": 500.0, "osm_ids": [101], "osm_section_indices": [1]}],
        "bus_xy_epsg7855": [[X0, Y0], [X0 + 1000, Y0], [X0 + 1500, Y0]], "crs": "EPSG:7855",
    },
    "model": {"std_type": "NA2XS2Y 1x185 RM/25 12/20 kV", "load_kw": 200.0, "power_factor": 0.95,
              "n_loads": 2, "load_buses": [1, 2]},
    "reference_limits": {"v_min_pu": 0.94, "loading_limit_pct": 100.0},
    "results": {
        "base": {"converged": True, "load_multiplier": 1.0, "total_load_kw": 400.0, "end_vm_pu": 0.9861,
                 "end_voltage_drop_pct": 1.39, "max_line_loading_pct": 27.39, "total_losses_kw": 1.55,
                 "bus_vm_pu": [1.0, 0.995, 0.9861], "bus_distance_km": [0.0, 1.0, 1.5],
                 "line_loading_pct": [27.4, 10.0]},
        "doubled": {"converged": True, "load_multiplier": 2.0, "total_load_kw": 800.0, "end_vm_pu": 0.9713,
                    "end_voltage_drop_pct": 2.87, "max_line_loading_pct": 55.72, "total_losses_kw": 6.2,
                    "bus_vm_pu": [1.0, 0.99, 0.9713], "line_loading_pct": [55.7, 20.0]}},
    "limits": {"voltage_multiplier": 4.03, "thermal_multiplier": 3.52,
               "binding": "thermal (loading > 100 % of cable rating)"},
}


def test_quality_counts_split_inside_vs_elsewhere():
    q = build.quality_counts(QUALITY)
    assert q["likely_error_inside_substations"] == 1 and q["likely_error_outside_substations"] == 1
    assert q["overlapping_lines_inside_substations"] == 1 and q["overlapping_lines_substation_sites"] == 1
    assert build.transformer_counts(QUALITY) == {
        "total": 5, "by_source": {"power=transformer": 3, "pole transformer tag": 2}}


def test_summarise_feeder():
    f = build.summarise_feeder(FEEDER)
    assert f["substation"] == "Zone A" and f["path_length_km"] == 1.5 and f["n_buses"] == 3
    assert f["base"]["end_voltage_pu"] == 0.986 and f["doubled"]["end_voltage_drop_pct"] == 2.87
    assert f["base"]["total_losses_kw"] == 1.6 and f["thermal_limit_multiplier"] == 3.52
    assert build.summarise_feeder(None) is None and build.summarise_feeder({}) is None


def test_build_summary_and_markdown():
    d = pd.DataFrame({"suburb_name": ["A"], "is_remainder": [False], "distance_km": [2.34],
                      "nearest_substation": "Zone A"})
    s = build.build_summary(_lines(), _subs(), 16, d, QUALITY, today=dt.date(2026, 1, 2),
                            duplicate_substation_notes=["same name: x"])
    assert s["generated"] == "2026-01-02" and s["zone_service_areas"] == 16
    assert s["quality_issues"]["by_group"]["likely_error"] == 2
    assert s["feeder"] is None and s["transformers"]["total"] == 5
    md = build.render_markdown(s)
    assert "2026-01-02" in md and "NOT Evoenergy network data" in md
    assert "km inside the ACT" in md and "Whole mapped ways incl. NSW sections" in md
    assert "Headline: 1.6 km of mapped line inside the ACT" in md
    assert "Substation bays and busbars" in md and "not an error count" in md
    assert "flagged by the checks" in md and "same name: x" in md
    assert "43" not in md and "3 from power=transformer" in md
    assert "Toy 11 kV feeder" not in md  # no feeder results -> section omitted
    s2 = build.build_summary(_lines(), _subs(), 16, d, QUALITY, FEEDER, today=dt.date(2026, 1, 2))
    md2 = build.render_markdown(s2)
    assert "Toy 11 kV feeder load flow (assumed loads and cable type; not a real Evoenergy feeder study)" in md2
    assert "feeder_results.md" in md2 and "0.986" in md2 and "0.971" in md2 and "3.52x" in md2
    json.dumps(s2)  # summary.json must be serialisable


def test_load_feeder_absent_or_bad(tmp_path):
    assert build.load_feeder(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert build.load_feeder(bad) is None


def test_missing_raw_exits_with_message(monkeypatch, capsys):
    monkeypatch.setattr(build, "missing_raw_files", lambda: [build.OUTPUTS_DIR / "nope.gpkg"])
    assert build.main() == 1
    assert "python -m act_network.fetch" in capsys.readouterr().out
