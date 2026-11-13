"""Unit tests for act_network.topology on small hand-made networks (EPSG:7855, metres)."""

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, box

from act_network import topology
from act_network.clean import PROJECT_CRS, line_class, parse_voltage

# Somewhere in Canberra in MGA zone 55
X0, Y0 = 690_000.0, 6_090_000.0


def lines_gdf(specs):
    """specs: list of (coords, voltage_tag) with coords relative to (X0, Y0)."""
    rows = []
    for i, (coords, vtag) in enumerate(specs):
        volts = parse_voltage(vtag)
        vmax = max(volts) if volts else None
        rows.append({
            "osm_id": 100 + i, "element": "way", "power": "minor_line", "voltage": vtag,
            "voltages": ";".join(map(str, volts)), "voltage_missing": not volts,
            "line_class": line_class(vmax),
            "geometry": LineString([(X0 + x, Y0 + y) for x, y in coords]),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECT_CRS)


def points_gdf(xy, start_id=900):
    return gpd.GeoDataFrame(
        {"osm_id": [start_id + i for i in range(len(xy))], "element": "node", "name": None},
        geometry=[Point(X0 + x, Y0 + y) for x, y in xy], crs=PROJECT_CRS)


def n_line_components(topo):
    return sum(1 for comp in nx.connected_components(topo.graph)
               if any(d["kind"] == "line" for _, _, d in topo.graph.edges(comp, data=True)))


# ---- voltage helpers --------------------------------------------------------

def test_voltage_set_parses_joined_string():
    assert topology.voltage_set("11000;415") == {11000, 415}
    assert topology.voltage_set("") == frozenset()
    assert topology.voltage_set(None) == frozenset()


def test_voltages_compatible_rules():
    vs = topology.voltage_set
    assert topology.voltages_compatible(vs("11000;415"), vs("415"))  # shared-pole LV
    assert not topology.voltages_compatible(vs("11000"), vs("415"))
    assert topology.voltages_compatible(vs(""), vs("132000"))  # unknown never conflicts


# ---- endpoint snapping ------------------------------------------------------

def test_endpoints_3m_apart_connect():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(103, 0), (200, 0)], "11000")])
    topo = topology.build_graph(lines, tol=5.0)
    assert n_line_components(topo) == 1


def test_endpoints_10m_apart_do_not_connect():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(110, 0), (200, 0)], "11000")])
    topo = topology.build_graph(lines, tol=5.0)
    assert n_line_components(topo) == 2


def test_short_way_does_not_merge_its_own_ends():
    # a 3 m way shorter than tol keeps two distinct end nodes
    lines = lines_gdf([([(0, 0), (3, 0)], "415")])
    topo = topology.build_graph(lines, tol=5.0)
    ep = topo.endpoints
    assert ep["node"].nunique() == 2


# ---- T-junctions --------------------------------------------------------------

def test_t_junction_onto_line_interior_connects_and_splits():
    # B ends 2 m from the middle of A (not near A's endpoints)
    lines = lines_gdf([([(0, 0), (200, 0)], "415"), ([(100, 50), (100, 2)], "415")])
    topo = topology.build_graph(lines, tol=5.0)
    assert n_line_components(topo) == 1
    # A was split into two edges at the junction
    a_edges = [d for _, _, d in topo.graph.edges(data=True) if d.get("line_idx") == 0]
    assert len(a_edges) == 2
    assert abs(sum(d["length_m"] for d in a_edges) - 200) < 1e-6
    b_end = topo.endpoints[(topo.endpoints.line_idx == 1) & (topo.endpoints.end == "end")].iloc[0]
    assert topo.line_degree_others(b_end["node"], 1) == 2


def test_t_junction_disabled_leaves_two_components():
    lines = lines_gdf([([(0, 0), (200, 0)], "415"), ([(100, 50), (100, 2)], "415")])
    topo = topology.build_graph(lines, tol=5.0, snap_interior=False)
    assert n_line_components(topo) == 2


def test_t_snap_between_transmission_and_distribution_is_blocked():
    lines = lines_gdf([([(0, 0), (500, 0)], "132000"), ([(250, 60), (250, 2)], "11000")])
    topo = topology.build_graph(lines, tol=5.0)
    assert n_line_components(topo) == 2
    assert len(topo.blocked_snaps) == 1
    topo2 = topology.build_graph(lines, tol=5.0, block_cross_class_tsnap=False)
    assert n_line_components(topo2) == 1


def test_shared_interior_vertex_joins_compatible_ways():
    # two ways cross through the same OSM node (X-junction), neither ends there
    lines = lines_gdf([([(0, 0), (50, 0), (100, 0)], "415"), ([(50, -50), (50, 0), (50, 50)], "415")])
    assert n_line_components(topology.build_graph(lines)) == 1
    # an 11 kV and an LV way merely sharing a pole are not joined
    lines2 = lines_gdf([([(0, 0), (50, 0), (100, 0)], "11000"), ([(50, -50), (50, 0), (50, 50)], "415")])
    assert n_line_components(topology.build_graph(lines2)) == 2


# ---- substations / transformers ----------------------------------------------

def test_endpoint_inside_footprint_links_substation():
    lines = lines_gdf([([(-200, 0), (10, 10)], "11000")])
    fp = gpd.GeoDataFrame({"osm_id": [500], "name": ["Zone"]},
                          geometry=[box(X0, Y0, X0 + 40, Y0 + 40)], crs=PROJECT_CRS)
    subs = gpd.GeoDataFrame({"osm_id": [500], "name": ["Zone"]},
                            geometry=[Point(X0 + 20, Y0 + 20)], crs=PROJECT_CRS)
    topo = topology.build_graph(lines, subs, fp)
    comps = topology.component_table(topo)
    assert comps["has_substation"].tolist() == [True]
    end = topo.endpoints[topo.endpoints.end == "end"].iloc[0]
    assert ("sub", 500) in end["device_links"]


def test_point_substation_within_sub_tol():
    lines = lines_gdf([([(0, 0), (100, 0)], "11000"), ([(1000, 0), (1100, 0)], "11000")])
    subs = points_gdf([(125, 0)])  # 25 m from line 0's end, far from line 1
    topo = topology.build_graph(lines, subs, None, sub_tol=30.0)
    by_line = {tuple(r.line_idx): r.has_substation for r in topology.component_table(topo).itertuples()}
    assert by_line == {(0,): True, (1,): False}


def test_transformer_connects_lv_to_11kv():
    # 11 kV way passes a pole-top transformer; an LV way starts 20 m away from it
    lines = lines_gdf([([(0, 0), (200, 0)], "11000"), ([(100, 20), (100, 150)], "415")])
    tx = points_gdf([(100, 0)], start_id=700)
    topo = topology.build_graph(lines, None, None, tx, tx_tol=30.0)
    assert n_line_components(topo) == 1
    comps = topology.component_table(topo)
    assert comps["has_transformer"].tolist() == [True]
    assert comps["has_substation"].tolist() == [False]  # transformers are not supply points


def test_component_table_orders_by_length():
    lines = lines_gdf([([(0, 0), (10, 0)], "415"), ([(500, 0), (900, 0)], "11000")])
    comps = topology.component_table(topology.build_graph(lines))
    assert comps["component_id"].tolist() == [0, 1]
    assert comps.loc[0, "length_m"] > comps.loc[1, "length_m"]
    assert comps.loc[0, "top_class"] == "distribution"


def test_runs_fast_on_a_few_thousand_lines():
    import time
    specs = [([(i * 20.0, 0), (i * 20.0 + 18.0, 0)], "415") for i in range(3000)]
    t = time.perf_counter()
    topo = topology.build_graph(lines_gdf(specs), tol=5.0)
    assert time.perf_counter() - t < 10
    assert n_line_components(topo) == 1


def test_small_footprint_also_uses_sub_tol_around_its_point():
    # 10 m x 10 m footprint; the line ends 15 m outside it (> footprint_tol) but
    # 20 m from the substation point (< sub_tol), as in the brief's rule
    fp = gpd.GeoDataFrame({"osm_id": [600], "name": [None]},
                          geometry=[box(X0 - 5, Y0 - 5, X0 + 5, Y0 + 5)], crs=PROJECT_CRS)
    subs = gpd.GeoDataFrame({"osm_id": [600], "name": [None]}, geometry=[Point(X0, Y0)], crs=PROJECT_CRS)
    lines = lines_gdf([([(20, 0), (300, 0)], "22000")])
    topo = topology.build_graph(lines, subs, fp, footprint_tol=5.0, sub_tol=30.0)
    assert topology.component_table(topo)["has_substation"].tolist() == [True]
    topo2 = topology.build_graph(lines, subs, fp, footprint_tol=5.0, sub_tol=10.0)
    assert topology.component_table(topo2)["has_substation"].tolist() == [False]
