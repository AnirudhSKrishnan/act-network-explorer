"""Unit tests for the data-quality checks in act_network.report (synthetic data, EPSG:7855)."""

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box

from act_network import report, topology
from act_network.clean import PROJECT_CRS, line_class, parse_voltage

X0, Y0 = 690_000.0, 6_090_000.0


def lines_gdf(specs):
    rows = []
    for i, (coords, vtag) in enumerate(specs):
        volts = parse_voltage(vtag)
        rows.append({
            "osm_id": 100 + i, "element": "way", "power": "minor_line", "voltage": vtag,
            "voltages": ";".join(map(str, volts)), "voltage_missing": not volts,
            "line_class": line_class(max(volts) if volts else None),
            "geometry": LineString([(X0 + x, Y0 + y) for x, y in coords]),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECT_CRS)


def subs_gdf(rows):
    """rows: list of (osm_id, name, (x, y))."""
    return gpd.GeoDataFrame(
        {"osm_id": [r[0] for r in rows], "element": "way", "name": [r[1] for r in rows]},
        geometry=[Point(X0 + r[2][0], Y0 + r[2][1]) for r in rows], crs=PROJECT_CRS)


NO_SUBS = subs_gdf([])


def dangling(lines, subs=None, fps=None, txs=None, boundary=None, **kw):
    topo = topology.build_graph(lines, subs, fps, txs, **kw)
    return report.find_dangling_ends(topo, boundary)


# ---- dangling ends ----------------------------------------------------------

def test_dangling_end_detected():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], "11000")])
    d = dangling(lines)
    # the shared middle node is not dangling; the two outer ends are
    assert len(d) == 2
    assert set(zip(d["line_idx"], d["end"])) == {(0, "start"), (1, "end")}


def test_t_junction_is_not_a_dangling_end():
    # B ends 2 m from A's interior: only A's two ends and B's far end dangle
    lines = lines_gdf([([(0, 0), (200, 0)], "415"), ([(100, 50), (100, 2)], "415")])
    d = dangling(lines)
    assert (1, "end") not in set(zip(d["line_idx"], d["end"]))
    assert len(d) == 3
    # without interior snapping the T-junction would be a false dangling end
    d_no = dangling(lines, snap_interior=False)
    assert (1, "end") in set(zip(d_no["line_idx"], d_no["end"]))


def test_near_miss_flagged_for_small_gap():
    lines = lines_gdf([([(0, 0), (100, 0)], "415"), ([(112, 0), (200, 0)], "415")])
    d = dangling(lines).set_index(["line_idx", "end"])
    assert bool(d.loc[(0, "end"), "near_miss"])
    assert abs(d.loc[(0, "end"), "nearest_m"] - 12) < 1e-6
    assert not bool(d.loc[(0, "start"), "near_miss"])


def test_end_at_transformer_is_not_dangling():
    lines = lines_gdf([([(0, 0), (100, 0)], "415")])
    txs = subs_gdf([(700, None, (120, 0))])  # 20 m from the end, within tx_tol
    d = dangling(lines, txs=txs, tx_tol=30.0)
    assert set(d["end"]) == {"start"}


def test_end_outside_study_area_is_marked():
    lines = lines_gdf([([(0, 0), (100, 0)], "132000")])
    boundary = box(X0 - 10, Y0 - 10, X0 + 50, Y0 + 10)
    d = dangling(lines, boundary=boundary).set_index("end")
    assert bool(d.loc["start", "inside_study_area"])
    assert not bool(d.loc["end", "inside_study_area"])


# ---- missing voltage ---------------------------------------------------------

def test_missing_voltage_flagged():
    lines = lines_gdf([([(0, 0), (100, 0)], None), ([(0, 50), (100, 50)], "11000")])
    assert report.find_missing_voltage(lines).tolist() == [True, False]
    # also without the precomputed flag (falls back to the parsed ``voltages`` string)
    assert report.find_missing_voltage(lines.drop(columns="voltage_missing")).tolist() == [True, False]


# ---- floating components ------------------------------------------------------

def _classified(lines, subs=None, fps=None, txs=None, boundary=None):
    topo = topology.build_graph(lines, subs, fps, txs)
    comps = report.classify_components(topology.component_table(topo), topo, boundary)
    return {tuple(r.line_idx): r.tier for r in comps.itertuples()}


def test_component_without_substation_is_floating():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(5000, 0), (5100, 0)], "11000")])
    fp = gpd.GeoDataFrame({"osm_id": [500], "name": ["Zone"]},
                          geometry=[box(X0 - 40, Y0 - 20, X0 + 2, Y0 + 20)], crs=PROJECT_CRS)
    subs = subs_gdf([(500, "Zone", (-20, 0))])
    tiers = _classified(lines, subs, fp)
    assert tiers == {(0,): "fed", (1,): "isolated"}


def test_component_within_30m_of_point_substation_is_fed():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000")])
    assert _classified(lines, subs_gdf([(501, None, (-25, 0))])) == {(0,): "fed"}
    assert _classified(lines, subs_gdf([(501, None, (-60, 0))])) == {(0,): "isolated"}


def test_transformer_only_and_boundary_tiers():
    lines = lines_gdf([([(0, 0), (100, 0)], "415"), ([(5000, 0), (5100, 0)], "415")])
    txs = subs_gdf([(700, None, (-10, 0))])
    boundary = box(X0 - 100, Y0 - 100, X0 + 5050, Y0 + 100)  # cuts through way 1
    assert _classified(lines, txs=txs, boundary=boundary) == {
        (0,): "transformer_only", (1,): "crosses_boundary"}


# ---- voltage conflicts ---------------------------------------------------------

def _conflicts(lines, **kw):
    return report.find_voltage_conflicts(topology.build_graph(lines, **kw))


def test_mixed_voltage_junction_without_substation_flagged():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], "415")])
    c = _conflicts(lines)
    assert len(c) == 1
    assert sorted(c.iloc[0]["line_idx"]) == [0, 1]


def test_multi_valued_tag_sharing_a_voltage_is_not_mixed():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000;415"), ([(100, 0), (200, 0)], "415")])
    assert _conflicts(lines).empty


def test_unknown_voltage_is_not_mixed():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], None)])
    assert _conflicts(lines).empty


def test_mixed_voltage_at_transformer_is_not_flagged():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], "415")])
    txs = subs_gdf([(700, None, (100, 3))])
    topo = topology.build_graph(lines, None, None, txs)
    assert report.find_voltage_conflicts(topo).empty


def test_mixed_voltage_t_junction_flagged():
    # LV way starting on the interior of an 11 kV way (unmapped pole-top transformer)
    lines = lines_gdf([([(0, 0), (200, 0)], "11000"), ([(100, 0), (100, 80)], "415")])
    assert len(_conflicts(lines)) == 1


# ---- duplicates / overlaps -------------------------------------------------------

def test_exact_duplicate_geometry_flagged():
    lines = lines_gdf([
        ([(0, 0), (50, 10), (100, 0)], "11000"),
        ([(100, 0), (50, 10), (0, 0)], "11000"),  # same geometry, reversed
        ([(0, 100), (100, 100)], "11000"),
    ])
    d = report.find_duplicate_lines(lines)
    assert len(d) == 1
    assert d.iloc[0]["line_idx"] == 1 and d.iloc[0]["original_idx"] == 0


def test_substantial_overlap_flagged_but_short_touch_is_not():
    lines = lines_gdf([
        ([(0, 0), (300, 0)], "132000"),
        ([(100, 0.3), (180, 0.3)], "132000"),  # 80 m lying on top of way 0
        ([(300, 0), (300, 200)], "132000"),  # just meets way 0's end
    ])
    over, n_partial = report.find_overlapping_lines(lines)
    assert len(over) == 1
    assert (over.iloc[0]["line_idx"], over.iloc[0]["other_idx"]) == (1, 0)
    assert over.iloc[0]["shared_m"] > 79


def test_duplicate_substation_same_name():
    subs = subs_gdf([(1, "Angle Crossing Zone Substation", (0, 0)),
                     (2, "angle crossing  zone substation", (500, 0)),
                     (3, "Other", (900, 0))])
    d = report.find_duplicate_substations(subs)
    assert d[["osm_id", "other_osm_id", "subtype"]].values.tolist() == [[2, 1, "same_name"]]


# ---- end to end on a tiny network ----------------------------------------------

def test_build_issues_and_outputs(tmp_path):
    lines = lines_gdf([
        ([(0, 0), (100, 0)], "11000"),       # fed from the substation at (-20, 0)
        ([(100, 0), (200, 0)], "415"),       # meets 11 kV directly -> voltage conflict
        ([(5000, 0), (5100, 0)], None),      # floating, untagged
        ([(5000, 0), (5100, 0)], None),      # exact duplicate of the above
    ])
    subs = subs_gdf([(500, "Zone", (-20, 0))])
    boundary = box(X0 - 1000, Y0 - 1000, X0 + 6000, Y0 + 1000)
    topo = topology.build_graph(lines, subs, None)
    issues, extra = report.build_issues(topo, subs, None, boundary)
    types = issues["issue_type"].value_counts().to_dict()
    assert types["voltage_conflict"] == 1
    assert types["missing_voltage"] == 2
    assert types["duplicate_line"] == 1
    assert types["floating_component"] == 1
    assert issues.crs.to_epsg() == PROJECT_CRS
    assert issues["osm_url"].str.startswith("https://www.openstreetmap.org/way/").all()
    assert issues[["lon", "lat"]].notna().all().all()
    assert issues["lon"].between(140, 160).all()  # WGS84 lon/lat
    # dangling ends: way 1's far end only (way 0 starts at the substation);
    # the duplicate pair's ends touch each other so they are not dangling
    dang = issues[issues["issue_type"] == "dangling_end"]
    assert dang["osm_id"].tolist() == [101]
    summary = report.summarise(topo, issues, extra)
    assert summary["checks"]["voltage_conflicts"]["junctions"] == 1
    assert summary["issues"]["total_rows"] == len(issues)
    json.dumps(summary, default=str)  # serialisable
    md = report.render_markdown(summary, issues)
    assert "not Evoenergy" in md and "OpenStreetMap contributors" in md


def test_run_writes_all_outputs(tmp_path):
    from act_network import clean

    gpkg = tmp_path / "act_power.gpkg"
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], "415"),
                       ([(5000, 0), (5100, 0)], None)])
    subs = subs_gdf([(500, "Zone", (-20, 0)), (501, "Sub", (3000, 0))])
    fps = gpd.GeoDataFrame({"osm_id": [501], "element": "way", "name": ["Sub"]},
                           geometry=[box(X0 + 2990, Y0 - 10, X0 + 3010, Y0 + 10)], crs=PROJECT_CRS)
    txs = subs_gdf([(700, None, (9000, 0))])
    boundary = gpd.GeoDataFrame(geometry=[box(X0 - 1000, Y0 - 1000, X0 + 10000, Y0 + 1000)], crs=PROJECT_CRS)
    for layer, gdf in [(clean.LINES_LAYER, lines), (clean.SUBSTATIONS_LAYER, subs),
                       (clean.FOOTPRINTS_LAYER, fps), (clean.TRANSFORMERS_LAYER, txs),
                       (clean.BOUNDARY_LAYER, boundary)]:
        gdf.to_file(gpkg, layer=layer, driver="GPKG")
    out = tmp_path / "outputs"
    topo_path = tmp_path / "topology.gpkg"
    summary = report.run(gpkg, out, topo_path)

    csv = pd.read_csv(out / "quality_issues.csv")
    for col in ["issue_type", "osm_id", "element", "line_class", "voltage", "details", "lon", "lat", "osm_url"]:
        assert col in csv.columns
    assert len(csv) == summary["issues"]["total_rows"]
    assert json.loads((out / "quality_summary.json").read_text())["issues"]["total_rows"] == len(csv)
    assert "OpenStreetMap" in (out / "quality_report.md").read_text()
    q = gpd.read_file(topo_path, layer=report.ISSUES_LAYER)
    assert q.crs.to_epsg() == PROJECT_CRS and len(q) == len(csv)
    comps = gpd.read_file(topo_path, layer=report.COMPONENTS_LAYER)
    assert len(comps) == summary["network"]["n_components"]


def test_lv_vs_11kv_conflict_is_to_check_but_22_vs_11_is_likely_error():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(100, 0), (200, 0)], "415"),
                       ([(0, 500), (100, 500)], "22000"), ([(100, 500), (200, 500)], "11000")])
    topo = topology.build_graph(lines)
    issues, _ = report.build_issues(topo, NO_SUBS, None)
    conf = issues[issues["issue_type"] == "voltage_conflict"].set_index("subtype")["group"].to_dict()
    assert conf == {"lv_vs_hv_distribution": "to_check", "distribution_vs_distribution": "likely_error"}


def test_overlap_inside_substation_gets_own_subtype_and_is_counted_separately():
    lines = lines_gdf([
        ([(0, 0), (80, 0)], "132000"),            # bay inside the footprint
        ([(10, 0.3), (50, 0.3)], "132000"),       # second bay lying on it
        ([(1000, 0), (1300, 0)], "132000"),       # open-network line
        ([(1100, 0.3), (1180, 0.3)], "132000"),   # overlapping it, outside any substation
    ])
    fp = gpd.GeoDataFrame({"osm_id": [500], "element": "way", "name": ["Sub"]},
                          geometry=[box(X0 - 10, Y0 - 10, X0 + 90, Y0 + 10)], crs=PROJECT_CRS)
    subs = subs_gdf([(500, "Sub", (40, 0))])
    topo = topology.build_graph(lines, subs, fp)
    issues, extra = report.build_issues(topo, subs, fp)
    ov = issues[issues["issue_type"] == "overlapping_lines"].set_index("osm_id")["subtype"].to_dict()
    assert ov == {101: "inside_substation/same_voltage", 103: "same_voltage"}
    s = report.summarise(topo, issues, extra)
    assert s["checks"]["overlapping_lines"]["inside_substations"] == 1
    assert s["checks"]["overlapping_lines"]["substation_sites"] == 1
    assert s["issues"]["likely_error_outside_substations"] == s["issues"]["by_group"]["likely_error"] - 1
    md = report.render_markdown(s, issues)
    assert "inside* substations" in md and "not substation-internal" in md
