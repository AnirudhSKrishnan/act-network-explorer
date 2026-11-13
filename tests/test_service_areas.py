"""Unit tests for act_network.service_areas using small hand-made data (no I/O).

All geometries are in EPSG:7855 (metres).
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from act_network import service_areas as sa

CRS = 7855
X0, Y0 = 690_000.0, 6_090_000.0


def _boundary(w=10_000.0, h=6_000.0):
    return gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + w, Y0 + h)], crs=CRS)


def _subs(rows):
    """rows: (osm_id, name, dx, dy, is_zone)."""
    return gpd.GeoDataFrame(
        {
            "osm_id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "is_zone": [r[4] for r in rows],
        },
        geometry=[Point(X0 + r[2], Y0 + r[3]) for r in rows],
        crs=CRS,
    )


THREE = [
    (1, "West Zone Substation", 1_000, 3_000, True),
    (2, "Mid Zone Substation", 5_000, 1_000, True),
    (3, "East Zone Substation", 9_000, 5_000, True),
]


def test_zone_substations_filters_and_dedupes_by_name():
    subs = _subs(
        THREE
        + [
            (4, "Not A Zone", 2_000, 2_000, False),
            (5, "Dup Zone Substation", 3_000, 3_000, True),
            (6, "Dup Zone Substation", 3_000, 5_000, True),
            (7, None, 4_000, 4_000, True),
            (8, None, 6_000, 4_000, True),
        ]
    )
    z = sa.zone_substations(subs)
    assert len(z) == 3 + 1 + 2  # 3 singles, 1 merged dup, 2 unnamed kept individually
    assert "Not A Zone" not in set(z["name"])
    dup = z[z["name"] == "Dup Zone Substation"].iloc[0]
    assert dup["osm_id"] == 5  # first id kept
    assert dup["osm_ids"] == "5;6"
    assert dup["n_merged"] == 2
    assert dup.geometry.equals_exact(Point(X0 + 3_000, Y0 + 4_000), 1e-6)  # centroid
    assert dup["merge_spread_m"] == pytest.approx(1_000.0)
    assert z[z["name"] == "West Zone Substation"].iloc[0]["osm_ids"] == "1"
    assert z["name"].isna().sum() == 2


def test_voronoi_count_equals_substation_count():
    z = sa.zone_substations(_subs(THREE))
    areas = sa.voronoi_areas(z, _boundary())
    assert len(areas) == len(z) == 3
    assert areas["name"].tolist() == z["name"].tolist()
    assert (areas["label"] == "rough nearest-substation area").all()
    assert areas["note"].str.contains("NOT a real feeder").all()


def test_voronoi_clipped_to_boundary_and_covers_it():
    b = _boundary()
    areas = sa.voronoi_areas(sa.zone_substations(_subs(THREE)), b)
    bound = b.geometry.iloc[0]
    for g in areas.geometry:
        assert bound.buffer(1e-6).contains(g)  # nothing outside the boundary
    union = areas.geometry.union_all()
    assert union.area == pytest.approx(bound.area, rel=1e-9)
    # cells tile: areas sum to the union (no overlaps)
    assert areas.geometry.area.sum() == pytest.approx(bound.area, rel=1e-9)
    assert areas["area_km2"].sum() == pytest.approx(bound.area / 1e6)


def test_each_seed_lies_in_its_own_polygon():
    z = sa.zone_substations(_subs(THREE))
    areas = sa.voronoi_areas(z, _boundary())
    for seed, cell in zip(z.geometry, areas.geometry):
        assert cell.contains(seed)
    assert areas["seed_in_boundary"].all()


def test_voronoi_single_point_gets_whole_boundary():
    z = sa.zone_substations(_subs(THREE[:1]))
    areas = sa.voronoi_areas(z, _boundary())
    assert len(areas) == 1
    assert areas["area_km2"].iloc[0] == pytest.approx(60.0)


def test_substation_outside_boundary_keeps_row_with_note():
    # Far outside, with the real substation between it and nothing: its cell
    # lies entirely outside the boundary -> empty after clipping, row retained.
    subs = _subs(THREE + [(9, "Far Zone Substation", 5_000, 500_000, True)])
    areas = sa.voronoi_areas(sa.zone_substations(subs), _boundary())
    assert len(areas) == 4
    far = areas[areas["name"] == "Far Zone Substation"].iloc[0]
    assert not far["seed_in_boundary"]
    assert far.geometry.is_empty and far["area_km2"] == 0
    assert "outside the ACT boundary" in far["note"]
    assert "empty" in far["note"]
    # the real ones still tile the boundary
    assert areas.geometry.union_all().area == pytest.approx(60e6, rel=1e-9)


def test_substation_just_outside_boundary_has_partial_cell():
    subs = _subs(THREE + [(9, "Edge Zone Substation", 5_000, -500, True)])
    areas = sa.voronoi_areas(sa.zone_substations(subs), _boundary())
    edge = areas[areas["name"] == "Edge Zone Substation"].iloc[0]
    assert not edge["seed_in_boundary"]
    assert edge["area_km2"] > 0 and not edge.geometry.is_empty


def test_voronoi_rejects_coincident_points_and_crs_mismatch():
    z = sa.zone_substations(_subs(THREE))
    dup = z.copy()
    dup.loc[1, "geometry"] = dup.geometry.iloc[0]
    with pytest.raises(ValueError):
        sa.voronoi_areas(dup, _boundary())
    with pytest.raises(ValueError):
        sa.voronoi_areas(z, _boundary().to_crs(4326))


def _suburbs():
    # Suburb squares 200 m wide; representative points sit at their centres.
    def sq(cx, cy):
        return box(X0 + cx - 100, Y0 + cy - 100, X0 + cx + 100, Y0 + cy + 100)

    return gpd.GeoDataFrame(
        {
            "SAL_CODE21": ["1", "2", "3", "4"],
            "suburb_name": ["Nearby", "Middling", "Faraway", "ACT Remainder - Far"],
            "is_remainder": [False, False, False, True],
        },
        geometry=[
            sq(1_100, 3_000),  # 100 m east of West sub
            sq(5_000, 2_500),  # 1500 m north of Mid sub
            sq(8_800, 1_000),  # 3800 m east of Mid sub (East sub is 4005 m away)
            sq(9_900, 300),  # remainder, furthest of all (but excluded from ranking)
        ],
        crs=CRS,
    )


def test_suburb_distance_ranking_hand_made():
    z = sa.zone_substations(_subs(THREE))
    areas = sa.voronoi_areas(z, _boundary())
    d = sa.suburb_distances(_suburbs(), z, areas)

    assert len(d) == 4  # all suburbs kept
    real = d[~d["is_remainder"]]
    assert real["suburb_name"].tolist() == ["Faraway", "Middling", "Nearby"]  # furthest first
    assert real["rank"].tolist() == [1, 2, 3]
    assert real["distance_km"].tolist() == pytest.approx([3.8, 1.5, 0.1])
    assert real["nearest_substation"].tolist() == [
        "Mid Zone Substation", "Mid Zone Substation", "West Zone Substation"]
    # remainder: listed (last), no rank, even though its distance is the largest
    rem = d[d["is_remainder"]].iloc[0]
    assert d.iloc[-1]["suburb_name"] == "ACT Remainder - Far"
    assert rem["rank"] is sa.pd.NA or sa.pd.isna(rem["rank"])
    assert rem["distance_km"] > 3.8
    # each suburb point falls in the rough area of its nearest substation
    assert (d["rough_area"] == d["nearest_substation"]).all()


def test_suburb_distances_uses_point_inside_concave_suburb():
    # L-shaped suburb whose centroid is outside the polygon; the point used
    # must still be inside it.
    from shapely.geometry import Polygon

    L = Polygon([(0, 0), (1000, 0), (1000, 50), (50, 50), (50, 1000), (0, 1000)])
    from shapely.affinity import translate

    L = translate(L, X0 + 3_000, Y0 + 2_000)
    assert not L.contains(L.centroid)
    suburbs = gpd.GeoDataFrame(
        {"suburb_name": ["L"], "is_remainder": [False]}, geometry=[L], crs=CRS)
    z = sa.zone_substations(_subs(THREE))
    d = sa.suburb_distances(suburbs, z)
    assert L.contains(d.geometry.iloc[0])
