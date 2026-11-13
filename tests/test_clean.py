"""Unit tests for act_network.clean using small synthetic data (no I/O)."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, box

from act_network import clean

# Around Canberra (EPSG:4326, lon/lat)
LON, LAT = 149.1, -35.3


def _raw_network(rows):
    """rows: list of dicts with power/geometry and optional tags."""
    base = {"element": "way", "voltage": None, "name": None, "operator": None,
            "substation": None, "cables": None, "circuits": None, "location": None}
    recs = [{**base, "osm_id": i + 1, **r} for i, r in enumerate(rows)]
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)


def _line(dx=0.01):
    return LineString([(LON, LAT), (LON + dx, LAT)])


def _sub_poly():
    return box(LON, LAT, LON + 0.001, LAT + 0.001)


# ---- parse_voltage ---------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("132000", [132000]),
    ("66000;11000", [66000, 11000]),
    ("11000;66000", [66000, 11000]),       # sorted descending
    ("11000;11000", [11000]),              # de-duplicated
    ("415", [415]),
    (" 132000 ; 11000 ", [132000, 11000]),  # whitespace tolerated
    (None, []),
    ("", []),
    ("   ", []),
    (float("nan"), []),
    ("unknown", []),
    ("11kV", [11000]),                     # unit honoured
    ("11 kV", [11000]),
    ("415 V", [415]),
    ("11000;unknown", [11000]),            # junk token dropped, rest kept
    ("0", []),
    ("-5", []),
])
def test_parse_voltage(raw, expected):
    assert clean.parse_voltage(raw) == expected


# ---- line_class / voltage_band ---------------------------------------------

@pytest.mark.parametrize("vmax, cls, band", [
    (330000, "transmission", "330 kV"),
    (132000, "transmission", "132 kV"),
    (66000, "transmission", "66 kV"),
    (22000, "distribution", "22 kV"),
    (11000, "distribution", "11 kV"),
    (415, "distribution", "LV (415 V)"),
    (33000, "distribution", "other"),
    (None, "unknown", "unknown"),
    (pd.NA, "unknown", "unknown"),
])
def test_class_and_band_scalars(vmax, cls, band):
    assert clean.line_class(vmax) == cls
    assert clean.voltage_band(vmax) == band


def test_clean_lines_columns_and_rules():
    raw = _raw_network([
        {"power": "line", "voltage": "330000", "geometry": _line()},
        {"power": "line", "voltage": "132000;11000", "geometry": _line()},
        {"power": "minor_line", "voltage": "11000;415", "geometry": _line()},
        {"power": "minor_line", "voltage": "415", "geometry": _line()},
        {"power": "minor_line", "voltage": None, "geometry": _line()},
        {"power": "cable", "voltage": "11000", "geometry": _line()},
        {"power": "minor_line", "voltage": "11000", "location": "underground", "geometry": _line()},
        {"power": "minor_line", "voltage": "11000", "location": "outdoor", "geometry": _line()},
        {"power": "substation", "geometry": Point(LON, LAT)},  # not a line: dropped
    ])
    out = clean.clean_lines(raw)
    assert list(out.columns) == clean.LINE_COLUMNS
    assert len(out) == 8
    assert out.crs.to_epsg() == 7855

    assert out["voltage_max"].dtype == "Int64"
    assert out["line_class"].tolist() == [
        "transmission", "transmission", "distribution", "distribution",
        "unknown", "distribution", "distribution", "distribution"]
    assert out["voltage_band"].tolist() == [
        "330 kV", "132 kV", "11 kV", "LV (415 V)", "unknown", "11 kV", "11 kV", "11 kV"]
    assert out["voltages"].tolist()[:3] == ["330000", "132000;11000", "11000;415"]
    assert out["voltages"].iloc[4] == ""
    assert out["voltage_missing"].tolist() == [False] * 4 + [True] + [False] * 3
    assert pd.isna(out["voltage_max"].iloc[4])
    # cable and location=underground are underground; outdoor is not
    assert out["is_underground"].tolist() == [False, False, False, False, False, True, True, False]
    # ~0.01 deg of longitude at -35.3 is roughly 900 m
    assert 800 < out["length_m"].iloc[0] < 1000


def test_clean_lines_explodes_multilinestrings_and_normalises_operator():
    multi = MultiLineString([_line(), _line(0.02)])
    raw = _raw_network([
        {"power": "line", "voltage": "132000", "operator": "ActewAGL", "geometry": multi},
        {"power": "line", "voltage": "330000", "operator": "Transgrid", "geometry": _line()},
    ])
    out = clean.clean_lines(raw)
    assert len(out) == 3
    assert set(out.geom_type) == {"LineString"}
    assert out["osm_id"].tolist() == [1, 1, 2]
    assert out["operator"].tolist() == ["Evoenergy", "Evoenergy", "TransGrid"]


@pytest.mark.parametrize("raw, expected", [
    ("Evoenergy", "Evoenergy"), ("evoenergy", "Evoenergy"), ("ActewAGL", "Evoenergy"),
    ("ACTEWAGL", "Evoenergy"), ("TransGrid", "TransGrid"), ("Transgrid", "TransGrid"),
    ("Essential Energy", "Essential Energy"), (None, None), ("", None),
])
def test_normalise_operator(raw, expected):
    assert clean.normalise_operator(raw) == expected


# ---- substations -----------------------------------------------------------

def _subs():
    return _raw_network([
        {"power": "substation", "name": "A Zone Substation", "voltage": "132000;11000",
         "substation": "distribution", "geometry": _sub_poly()},          # zone
        {"power": "substation", "name": "Plain", "voltage": "132000;11000",
         "substation": "transmission", "geometry": _sub_poly()},          # zone by voltage
        {"power": "substation", "name": "Canberra Substation", "voltage": "330000;132000",
         "geometry": _sub_poly()},                                         # transmission
        {"power": "substation", "name": "Queanbeyan STS", "voltage": "132000;66000",
         "geometry": _sub_poly()},                                         # transmission by name
        {"power": "substation", "name": "Foo Terminal Station", "voltage": None,
         "geometry": _sub_poly()},                                         # transmission by name
        {"power": "substation", "name": "Bruce Switch Station", "voltage": None,
         "geometry": _sub_poly()},                                         # zone by name
        {"power": "substation", "name": None, "voltage": "132000",
         "geometry": _sub_poly()},                                         # one high voltage only
        {"power": "substation", "name": None, "voltage": None, "substation": "minor_distribution",
         "geometry": Point(LON, LAT)},                                     # untagged point
        {"power": "transformer", "geometry": Point(LON, LAT)},            # ignored
    ])


def test_substation_classification():
    out = clean.clean_substations(_subs())
    assert list(out.columns) == clean.SUBSTATION_COLUMNS
    assert len(out) == 8
    assert out["sub_class"].tolist() == [
        "zone", "zone", "transmission", "transmission", "transmission",
        "zone", "distribution", "distribution"]
    assert out["is_zone"].tolist() == [True, True, False, False, False, True, False, False]
    assert bool(out["voltage_missing"].iloc[-1])


def test_substations_are_points_with_original_geom_type():
    out = clean.clean_substations(_subs())
    assert out.crs.to_epsg() == 7855
    assert set(out.geom_type) == {"Point"}
    assert out["osm_geom_type"].tolist() == ["Polygon"] * 7 + ["Point"]
    # polygon -> centroid, so it lies within the (projected) footprint
    foot = clean.clean_substation_footprints(_subs())
    assert len(foot) == 7
    assert set(foot.geom_type) == {"Polygon"}
    assert foot.crs.to_epsg() == 7855
    assert foot.geometry.iloc[0].contains(out.geometry.iloc[0])


def test_substation_class_direct():
    assert clean.substation_class("x", [330000, 132000]) == "transmission"
    assert clean.substation_class("Canberra STS", []) == "transmission"
    assert clean.substation_class("x", [132000, 11000]) == "zone"
    assert clean.substation_class("x", [66000, 11000]) == "zone"
    assert clean.substation_class("x", [11000, 415]) == "distribution"
    assert clean.substation_class(None, []) == "distribution"
    assert clean.substation_class("Stsomething", []) == "distribution"  # STS needs word boundary


def test_duplicate_substations_not_dropped():
    dup = _raw_network([
        {"power": "substation", "name": "Angle Crossing Zone Substation",
         "voltage": "132000;11000", "geometry": _sub_poly()},
        {"power": "substation", "name": "Angle Crossing Zone Substation",
         "voltage": "132000;11000", "geometry": _sub_poly()},
    ])
    assert len(clean.clean_substations(dup)) == 2


# ---- transformers / supports / boundary -------------------------------------

def test_transformers_supports_boundary():
    raw = _raw_network([
        {"power": "transformer", "operator": "ACTEWAGL", "geometry": Point(LON, LAT)},
        {"power": "line", "geometry": _line()},
    ])
    tr = clean.clean_transformers(raw)
    assert list(tr.columns) == ["osm_id", "element", "name", "operator", "source", "geometry"]
    assert len(tr) == 1 and tr["operator"].iloc[0] == "Evoenergy"
    assert tr["source"].tolist() == ["power=transformer"]

    sup = gpd.GeoDataFrame({"osm_id": [1, 2], "power": ["pole", "tower"]},
                           geometry=[Point(LON, LAT), Point(LON + 0.1, LAT)], crs=4326)
    out = clean.clean_supports(sup)  # raw without element/transformer columns still works
    assert list(out.columns) == ["osm_id", "element", "power", "transformer", "geometry"]
    assert out["element"].tolist() == ["node", "node"]
    assert out["transformer"].isna().all()
    assert out.crs.to_epsg() == 7855

    b = gpd.GeoDataFrame({"name": ["a", "b"]},
                         geometry=[box(LON, LAT, LON + 1, LAT + 1), box(LON + 0.5, LAT, LON + 2, LAT + 1)],
                         crs=4326)
    ob = clean.clean_boundary(b)
    assert len(ob) == 1 and ob.crs.to_epsg() == 7855 and ob.geom_type.iloc[0] == "Polygon"


# ---- suburbs ---------------------------------------------------------------

def test_clean_suburbs():
    raw = gpd.GeoDataFrame(
        {"SAL_CODE21": [80001, 80002, 80003],
         "SAL_NAME21": ["Acton (ACT)", "Ainslie", "ACT Remainder - Belconnen"],
         "AREASQKM21": [1.0, 2.0, 3.0]},
        geometry=[box(LON + i * 0.01, LAT, LON + (i + 1) * 0.01, LAT + 0.01) for i in range(3)],
        crs=7844)
    out = clean.clean_suburbs(raw)
    assert list(out.columns) == ["SAL_CODE21", "suburb_name", "is_remainder", "geometry"]
    assert out["suburb_name"].tolist() == ["Acton", "Ainslie", "ACT Remainder - Belconnen"]
    assert out["is_remainder"].tolist() == [False, False, True]
    assert out.crs.to_epsg() == 7855


def test_strip_act_suffix():
    assert clean.strip_act_suffix("Acton (ACT)") == "Acton"
    assert clean.strip_act_suffix("Ainslie") == "Ainslie"
    assert clean.strip_act_suffix("Hall (ACT) ") == "Hall"


def test_constants():
    assert clean.LINES_LAYER == "lines"
    assert clean.PROCESSED_PATH.name == "act_power.gpkg"


def test_pole_top_transformers_join_transformers_layer():
    raw = _raw_network([{"power": "transformer", "geometry": Point(LON, LAT)}])
    sup = gpd.GeoDataFrame(
        {"element": "node", "osm_id": [11, 12, 13], "power": ["pole", "pole", "tower"],
         "operator": ["evoenergy", None, None], "transformer": ["distribution", None, None]},
        geometry=[Point(LON + 0.01, LAT), Point(LON + 0.02, LAT), Point(LON + 0.03, LAT)], crs=4326)
    tr = clean.clean_transformers(raw, sup)
    assert tr["osm_id"].tolist() == [1, 11]  # untagged pole and tower excluded
    assert tr["source"].tolist() == ["power=transformer", "pole transformer tag"]
    assert tr["element"].tolist() == ["way", "node"]
    assert tr["operator"].iloc[1] == "Evoenergy"
    assert tr.crs.to_epsg() == 7855 and set(tr.geom_type) == {"Point"}
    # supports without a transformer column (older raw cache) add nothing
    assert len(clean.clean_transformers(raw, sup.drop(columns="transformer"))) == 1


def test_length_in_act_and_substation_in_act():
    # ACT stand-in: the western half of a 0.01-degree line
    act = gpd.GeoDataFrame(geometry=[box(LON - 0.05, LAT - 0.001, LON + 0.005, LAT + 0.001)], crs=4326)
    raw = _raw_network([
        {"power": "line", "voltage": "132000", "geometry": _line(0.01)},            # half inside
        {"power": "line", "voltage": "132000", "geometry": _line(-0.01)},           # all inside
        {"power": "line", "voltage": "132000",
         "geometry": LineString([(LON + 0.02, LAT), (LON + 0.03, LAT)])},           # all outside
    ])
    out = clean.clean_lines(raw, act)
    assert len(out) == 3  # nothing dropped
    frac = (out["length_in_act_m"] / out["length_m"]).tolist()
    assert frac == pytest.approx([0.5, 1.0, 0.0], abs=1e-3)
    # no boundary -> whole length
    assert (clean.clean_lines(raw)["length_in_act_m"] == clean.clean_lines(raw)["length_m"]).all()

    subs = _raw_network([
        {"power": "substation", "name": "In", "geometry": Point(LON, LAT)},
        {"power": "substation", "name": "Out", "geometry": box(LON + 0.02, LAT, LON + 0.021, LAT + 0.001)},
    ])
    s = clean.clean_substations(subs, act)
    assert s["in_act"].tolist() == [True, False]
    assert len(s) == 2
    assert clean.clean_substation_footprints(subs, act)["in_act"].tolist() == [False]


def test_bay_and_busbar_flagged_as_substation_internal():
    raw = _raw_network([
        {"power": "line", "voltage": "132000", "line": "bay", "geometry": _line()},
        {"power": "line", "voltage": "132000", "line": "busbar", "geometry": _line()},
        {"power": "line", "voltage": "132000", "line": None, "geometry": _line()},
    ])
    out = clean.clean_lines(raw)
    assert out["is_substation_internal"].tolist() == [True, True, False]
    assert out["line"].tolist()[:2] == ["bay", "busbar"]
    # raw data without a line column still works
    assert not clean.clean_lines(raw.drop(columns="line"))["is_substation_internal"].any()
