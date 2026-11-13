"""Offline tests for act_network.feeder (synthetic graphs and toy pandapower feeders)."""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from act_network import feeder, topology
from act_network.clean import PROJECT_CRS, line_class, parse_voltage

X0, Y0 = 690_000.0, 6_090_000.0  # somewhere in Canberra, MGA zone 55


def lines_gdf(specs):
    """specs: list of (osm_id, coords relative to (X0, Y0), voltage tag)."""
    rows = []
    for oid, coords, vtag in specs:
        volts = parse_voltage(vtag)
        rows.append({
            "osm_id": oid, "power": "minor_line", "voltage": vtag,
            "voltages": ";".join(map(str, volts)), "line_class": line_class(max(volts) if volts else None),
            "geometry": LineString([(X0 + x, Y0 + y) for x, y in coords]),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECT_CRS)


def subs_gdf(specs):
    """specs: list of (osm_id, (x, y))."""
    return gpd.GeoDataFrame({"osm_id": [o for o, _ in specs], "name": None},
                            geometry=[Point(X0 + x, Y0 + y) for _, (x, y) in specs], crs=PROJECT_CRS)


# ---- path selection --------------------------------------------------------

def test_path_selection_follows_longest_11kv_path_only():
    lines = lines_gdf([
        (100, [(0, 10), (0, 1000)], "11000"),       # trunk, part 1
        (101, [(0, 1000), (0, 2500)], "11000"),     # trunk, part 2 -> expected end
        (102, [(0, 1000), (500, 1000)], "11000"),   # shorter 11 kV branch
        (103, [(10, 0), (5000, 0)], "415"),         # longer LV line off the substation: ignore
        (104, [(-10, 0), (-6000, 0)], "132000"),    # longer 132 kV line: ignore
        (105, [(3000, 3010), (3000, 3500)], "11000"),  # another zone sub's short feeder
    ])
    subs = subs_gdf([(900, (0, 0)), (901, (3000, 3000))])
    topo = topology.build_graph(lines, subs)
    sel = feeder.select_feeder_path(topo.graph, [("sub", 900), ("sub", 901)])
    assert not sel["fallback"]
    assert sel["substation_node"] == ("sub", 900)
    assert [s["osm_id"] for s in sel["sections"]] == [100, 101]
    assert sel["total_length_m"] == pytest.approx(2490, abs=1)
    assert sel["n_zone_substations_with_11kv"] == 2
    assert len(sel["branch_nodes"]) == 1  # the T at (0, 1000)


def test_path_selection_does_not_pass_through_another_substation():
    lines = lines_gdf([
        (100, [(0, 10), (0, 1000)], "11000"),
        (101, [(0, 1000), (0, 2500)], "11000"),
        (102, [(0, 10), (300, 10)], "11000"),
    ])
    subs = subs_gdf([(900, (0, 0)), (950, (0, 1000))])  # 950: another substation on the trunk
    topo = topology.build_graph(lines, subs)
    sel = feeder.select_feeder_path(topo.graph, [("sub", 900)])
    assert [s["osm_id"] for s in sel["sections"]] == [102]
    assert sel["total_length_m"] == pytest.approx(300, abs=1)


def test_path_selection_falls_back_when_no_zone_substation_touches_11kv():
    lines = lines_gdf([
        (100, [(0, 0), (800, 0)], "11000"),
        (101, [(800, 0), (800, 900)], "11000"),
        (102, [(5000, 0), (5400, 0)], "11000"),
    ])
    subs = subs_gdf([(900, (9000, 9000))])
    topo = topology.build_graph(lines, subs)
    sel = feeder.select_feeder_path(topo.graph, [("sub", 900)])
    assert sel["fallback"] and sel["substation_node"] is None
    assert sorted(s["osm_id"] for s in sel["sections"]) == [100, 101]
    assert sel["total_length_m"] == pytest.approx(1700, abs=1)


def test_merge_short_sections():
    assert feeder.merge_short_sections([202, 124, 2283, 113]) == [[0, 1], [2, 3]]
    assert feeder.merge_short_sections([50, 500, 600]) == [[0, 1], [2]]
    assert feeder.merge_short_sections([300, 400]) == [[0], [1]]
    assert feeder.merge_short_sections([50]) == [[0]]


# ---- pandapower toy feeders ------------------------------------------------

def test_three_bus_feeder_converges():
    net = feeder.build_feeder_net([1.0, 1.0])
    assert len(net.bus) == 3 and len(net.line) == 2 and len(net.load) == 2
    res = feeder.run_load_flow(net)
    assert res["converged"]
    assert res["bus_vm_pu"][0] == pytest.approx(feeder.SOURCE_VM_PU)
    assert 0 < res["end_voltage_drop_pct"] < 1
    assert res["total_load_kw"] == pytest.approx(400)
    assert res["total_losses_kw"] > 0


def test_voltage_decreases_monotonically_along_loaded_radial_feeder():
    net = feeder.build_feeder_net([0.5, 1.2, 0.8, 2.0, 0.7])  # loads on every bus incl. the last
    res = feeder.run_load_flow(net)
    assert res["converged"]
    assert np.all(np.diff(res["bus_vm_pu"]) < 0)
    assert res["min_vm_bus"] == res["end_bus"]
    # current falls along the feeder as load is shed at each bus
    assert np.all(np.diff(res["line_loading_pct"]) < 0)


def test_doubling_load_increases_voltage_drop():
    net = feeder.build_feeder_net([1.0, 2.0, 1.5])
    base = feeder.run_load_flow(net, 1.0)
    dbl = feeder.run_load_flow(net, 2.0)
    assert dbl["end_voltage_drop_pct"] > base["end_voltage_drop_pct"]
    assert 1.8 < dbl["end_voltage_drop_pct"] / base["end_voltage_drop_pct"] < 2.3
    assert dbl["max_line_loading_pct"] > base["max_line_loading_pct"]
    assert dbl["total_losses_kw"] > 3.5 * base["total_losses_kw"]  # ~I^2


def test_voltage_drop_matches_hand_calculation():
    # one 5 km section, 2 MW at 0.95 PF at the end: dV ~ (P R + Q X) / V^2
    net = feeder.build_feeder_net([5.0], load_kw=2000)
    res = feeder.run_load_flow(net)
    p, q = 2.0, 2.0 * math.tan(math.acos(0.95))
    approx_pct = (p * 0.161 * 5 + q * 0.117 * 5) / 11.0**2 * 100
    assert res["end_voltage_drop_pct"] == pytest.approx(approx_pct, rel=0.15)


def test_find_limit_multiplier_brackets_the_voltage_limit():
    net = feeder.build_feeder_net([2.0, 2.0, 2.0])
    m = feeder.find_limit_multiplier(net, feeder.voltage_violated)
    assert m is not None and m > 1
    assert feeder.voltage_violated(feeder.run_load_flow(net, m))
    assert not feeder.voltage_violated(feeder.run_load_flow(net, m - 0.02))
    assert (net.load.scaling == 1.0).all()


def test_unknown_std_type_raises():
    with pytest.raises(ValueError):
        feeder.build_feeder_net([1.0], std_type="no such cable")
