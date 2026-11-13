"""Rough nearest-substation areas (Voronoi) and suburb-to-substation distances.

Run with::

    python -m act_network.service_areas

HONESTY NOTE -- read before quoting any number from here
--------------------------------------------------------
The polygons produced here are **rough nearest-substation areas**: every point
in a polygon is geometrically closer to that zone substation than to any other
zone substation (a Voronoi diagram, clipped to the ACT boundary). They are a
geometric estimate only. They are NOT real feeder boundaries, NOT Evoenergy's
supply areas, and they ignore the actual line routes, switching arrangements,
capacity and load. The data is public OpenStreetMap (c) OpenStreetMap
contributors (ODbL), NOT Evoenergy's network data, and OSM power mapping is
incomplete (a missing zone substation in OSM silently enlarges its neighbours'
areas). Suburb boundaries: ABS ASGS 2021 (CC BY 4.0).

Design choices
--------------
* "Zone substations" are the ``is_zone`` rows of the cleaned ``substations``
  layer. Same-name duplicates (e.g. one substation mapped as two OSM ways) are
  merged into one seed at the centroid of the duplicates, so a substation does
  not get two cells.
* The Voronoi diagram is built with ``shapely.voronoi_polygons(...,
  ordered=True)`` so output cell ``i`` belongs to input point ``i`` (no
  spatial join back to seeds is needed), extended well beyond the boundary so
  cells cover it, then intersected with the boundary.
* Distances to suburbs use each suburb's **representative point**
  (``representative_point()``), not its centroid: the centroid of a concave or
  very irregular polygon (e.g. the "ACT Remainder" rural areas) can fall
  outside the polygon, whereas the representative point is guaranteed inside.
  Distances are straight-line (not along the network) in EPSG:7855 metres,
  reported in km.

The functions are pure (GeoDataFrame in, GeoDataFrame out) so they can be unit
tested; ``main()`` does the file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point, box

from act_network.clean import (
    BOUNDARY_LAYER,
    PROCESSED_DIR,
    PROCESSED_PATH,
    PROJECT_CRS,
    SUBSTATIONS_LAYER,
    SUBURBS_LAYER,
)

SERVICE_AREAS_PATH = PROCESSED_DIR / "service_areas.gpkg"
AREAS_LAYER = "zone_service_areas"
ZONE_SUBS_LAYER = "zone_substations"
SUBURB_DISTANCES_LAYER = "suburb_distances"

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
SUBURB_DISTANCES_CSV = OUTPUTS_DIR / "suburb_distances.csv"
SUMMARY_JSON = OUTPUTS_DIR / "service_areas_summary.json"

AREA_LABEL = "rough nearest-substation area"
AREA_NOTE = (
    "Rough nearest-substation area: a geometric (Voronoi) estimate from public "
    "OpenStreetMap data. NOT a real feeder or supply boundary; not Evoenergy data."
)
TOP_N = 10


# ---------------------------------------------------------------------------
# 1. Zone substations
# ---------------------------------------------------------------------------


def zone_substations(subs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return one point per zone substation (``is_zone`` rows, de-duplicated by name).

    * Rows with the same ``name`` are merged into a single point at the centroid
      of the duplicates. The first row's attributes (and its ``osm_id``) are
      kept; every member's id is recorded in the ``osm_ids`` column
      (";"-joined, first id first).
    * ``n_merged`` is the number of source rows; ``merge_spread_m`` is the
      largest distance (m) from the merged point to any member, so a reviewer
      can see whether "duplicates" were really the same place (0 for singles).
    * Unnamed zone substations (null / blank name) are never merged.
    * Output order follows first appearance in the input.
    """
    zone = subs[subs["is_zone"].fillna(False).astype(bool)].copy()
    cols = [c for c in zone.columns if c not in ("geometry",)]
    names = zone["name"] if "name" in zone.columns else pd.Series([None] * len(zone), index=zone.index)
    zone["_key"] = [
        f"name:{n.strip()}" if isinstance(n, str) and n.strip() else f"row:{i}"
        for i, n in enumerate(names)
    ]

    rows = []
    for _, grp in zone.groupby("_key", sort=False):
        first = grp.iloc[0]
        pts = list(grp.geometry)
        merged = pts[0] if len(pts) == 1 else MultiPoint(pts).centroid
        rec = {c: first[c] for c in cols}
        rec["osm_ids"] = ";".join(str(i) for i in grp["osm_id"])
        rec["n_merged"] = len(grp)
        rec["merge_spread_m"] = float(max(merged.distance(p) for p in pts)) if len(pts) > 1 else 0.0
        rec["geometry"] = merged
        rows.append(rec)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=subs.crs)
    if out.empty:
        out = gpd.GeoDataFrame(
            {**{c: [] for c in cols}, "osm_ids": [], "n_merged": [], "merge_spread_m": []},
            geometry=[], crs=subs.crs,
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Voronoi areas
# ---------------------------------------------------------------------------


def voronoi_areas(points: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Voronoi cells around ``points`` clipped to ``boundary``: one row per point.

    Output has the input's ``name`` / ``osm_id`` (when present), ``seed_in_boundary``,
    ``area_km2``, ``label`` ("rough nearest-substation area"), ``note`` and the
    clipped polygon geometry. These are rough geometric areas, not feeder
    boundaries (see the module docstring).

    A point lying outside the boundary still has a cell (its neighbours'
    cells are shrunk accordingly, which is the right geometry). The cell is
    clipped to the boundary; if nothing is left the row is KEPT with an empty
    geometry, ``area_km2`` 0 and an explanatory note -- rows are never dropped,
    so output count == input count. ``seed_in_boundary`` flags such seeds.
    """
    if len(points) == 0:
        raise ValueError("voronoi_areas needs at least one point")
    if points.crs != boundary.crs:
        raise ValueError(f"CRS mismatch: points {points.crs} vs boundary {boundary.crs}")
    if points.geometry.duplicated().any():
        raise ValueError("coincident seed points; de-duplicate (zone_substations) first")

    bound = boundary.geometry.union_all()
    seeds = list(points.geometry)

    if len(seeds) == 1:
        cells = [bound]
    else:
        minx, miny, maxx, maxy = bound.union(MultiPoint(seeds)).bounds
        pad = max(maxx - minx, maxy - miny)  # generous: cells reach well past the boundary
        extent = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
        # ordered=True: output cell i corresponds to input point i.
        diagram = shapely.voronoi_polygons(MultiPoint(seeds), extend_to=extent, ordered=True)
        cells = list(diagram.geoms)
        if len(cells) != len(seeds):  # pragma: no cover - defensive
            raise RuntimeError(f"Voronoi returned {len(cells)} cells for {len(seeds)} seeds")

    clipped = [c.intersection(bound) for c in cells]
    in_bound = [bound.covers(p) for p in seeds]

    out = gpd.GeoDataFrame(index=points.index, geometry=clipped, crs=points.crs)
    for col in ("osm_id", "osm_ids", "name"):
        if col in points.columns:
            out[col] = points[col].to_numpy()
    out["seed_in_boundary"] = in_bound
    out["area_km2"] = out.geometry.area / 1e6
    out["label"] = AREA_LABEL
    notes = []
    for empty, inside in zip(out.geometry.is_empty, in_bound):
        parts = [AREA_NOTE]
        if not inside:
            parts.append("Substation lies outside the ACT boundary.")
        if empty:
            parts.append("Clipped cell is empty (no area inside the boundary).")
        notes.append(" ".join(parts))
    out["note"] = notes
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Suburb distances
# ---------------------------------------------------------------------------


def suburb_distances(
    suburbs: gpd.GeoDataFrame,
    zone_subs: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Distance (km) from each suburb's representative point to the nearest zone substation.

    Geometry of the result is the representative point used (guaranteed to lie
    inside its suburb, unlike the centroid). Columns: suburb identifiers,
    ``is_remainder``, ``distance_km``, ``nearest_substation`` (+ its
    ``nearest_osm_id``), ``rough_area`` (name of the rough nearest-substation
    area that contains the point, if ``areas`` is given) and ``rank``.

    ``rank`` is 1 = furthest, computed among REAL suburbs only (rows with
    ``is_remainder`` False; ties share the lowest rank); remainder rows get
    ``<NA>`` so "ACT Remainder - X" never appears in the ranking. All suburbs are
    returned, sorted by rank (real, furthest first) then remainder rows by
    distance descending.
    """
    if suburbs.crs != zone_subs.crs:
        raise ValueError(f"CRS mismatch: suburbs {suburbs.crs} vs substations {zone_subs.crs}")
    if len(zone_subs) == 0:
        raise ValueError("suburb_distances needs at least one zone substation")

    pts = suburbs.geometry.representative_point().to_numpy()
    sub_pts = zone_subs.geometry.to_numpy()
    dist = shapely.distance(pts[:, None], sub_pts[None, :])  # metres, (n_suburbs, n_subs)
    nearest = dist.argmin(axis=1)  # first index wins ties

    keep = [c for c in ("SAL_CODE21", "suburb_name", "is_remainder") if c in suburbs.columns]
    out = gpd.GeoDataFrame(suburbs[keep].copy(), geometry=pts, crs=suburbs.crs)
    out["is_remainder"] = out["is_remainder"].fillna(False).astype(bool)
    out["distance_km"] = dist[np.arange(len(pts)), nearest] / 1000.0
    out["nearest_substation"] = zone_subs["name"].to_numpy()[nearest]
    if "osm_id" in zone_subs.columns:
        out["nearest_osm_id"] = zone_subs["osm_id"].to_numpy()[nearest]

    out["rough_area"] = None
    if areas is not None:
        hit = gpd.sjoin(out[["geometry"]], areas[["name", "geometry"]], how="left", predicate="intersects")
        hit = hit[~hit.index.duplicated(keep="first")]  # point exactly on a shared edge
        out["rough_area"] = hit["name"].reindex(out.index).to_numpy()

    real = ~out["is_remainder"]
    out["rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[real, "rank"] = (
        out.loc[real, "distance_km"].rank(method="min", ascending=False).astype("Int64")
    )
    out["_remainder_order"] = out["is_remainder"].astype(int)
    out = out.sort_values(["_remainder_order", "distance_km"], ascending=[True, False], kind="stable")
    return out.drop(columns="_remainder_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary + I/O
# ---------------------------------------------------------------------------


def build_summary(
    subs: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    dists: gpd.GeoDataFrame,
    top_n: int = TOP_N,
) -> dict:
    """JSON-able summary of the run (counts, per-substation areas, furthest suburbs)."""
    real = dists[~dists["is_remainder"]].sort_values("rank")
    merged = zones[zones["n_merged"] > 1]
    mismatch = dists[dists["rough_area"].notna() & (dists["rough_area"] != dists["nearest_substation"])]
    return {
        "label": AREA_LABEL,
        "disclaimer": AREA_NOTE
        + " Distances are straight-line from a suburb's representative point, not along the network.",
        "data_source": "Public OpenStreetMap (c) OpenStreetMap contributors (ODbL); suburbs ABS ASGS 2021 (CC BY 4.0). Not Evoenergy network data.",
        "crs": f"EPSG:{PROJECT_CRS}",
        "counts": {
            "substations_total": int(len(subs)),
            "zone_substation_rows": int(subs["is_zone"].sum()),
            "zone_substations_after_dedupe": int(len(zones)),
            "service_areas": int(len(areas)),
            "empty_areas": int(areas.geometry.is_empty.sum()),
            "substations_outside_boundary": int((~areas["seed_in_boundary"]).sum()),
            "suburbs_total": int(len(dists)),
            "suburbs_real": int((~dists["is_remainder"]).sum()),
            "suburbs_remainder": int(dists["is_remainder"].sum()),
            "suburb_points_in_a_different_area_than_nearest": int(len(mismatch)),
        },
        "merged_duplicates": [
            {"name": r["name"], "osm_ids": r["osm_ids"], "merge_spread_m": round(r["merge_spread_m"], 1)}
            for _, r in merged.iterrows()
        ],
        "total_area_km2": round(float(areas["area_km2"].sum()), 2),
        "areas_km2": {
            str(r["name"]): round(float(r["area_km2"]), 2)
            for _, r in areas.sort_values("area_km2", ascending=False).iterrows()
        },
        "top_furthest_real_suburbs": [
            {
                "rank": int(r["rank"]),
                "suburb": r["suburb_name"],
                "distance_km": round(float(r["distance_km"]), 2),
                "nearest_substation": r["nearest_substation"],
            }
            for _, r in real.head(top_n).iterrows()
        ],
    }


def main() -> None:
    subs = gpd.read_file(PROCESSED_PATH, layer=SUBSTATIONS_LAYER)
    boundary = gpd.read_file(PROCESSED_PATH, layer=BOUNDARY_LAYER)
    suburbs = gpd.read_file(PROCESSED_PATH, layer=SUBURBS_LAYER)

    zones = zone_substations(subs)
    areas = voronoi_areas(zones, boundary)
    dists = suburb_distances(suburbs, zones, areas)

    SERVICE_AREAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_AREAS_PATH.unlink(missing_ok=True)  # fresh file; never touches act_power.gpkg
    areas.to_file(SERVICE_AREAS_PATH, layer=AREAS_LAYER, driver="GPKG")
    zones.to_file(SERVICE_AREAS_PATH, layer=ZONE_SUBS_LAYER, driver="GPKG")
    dists.to_file(SERVICE_AREAS_PATH, layer=SUBURB_DISTANCES_LAYER, driver="GPKG")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = dists.drop(columns="geometry").rename(columns={"rank": "rank_among_real_suburbs"})
    csv["distance_km"] = csv["distance_km"].round(3)
    csv = csv[["suburb_name", "distance_km", "nearest_substation", "is_remainder",
               "rank_among_real_suburbs", "rough_area"]]
    csv.to_csv(SUBURB_DISTANCES_CSV, index=False)

    summary = build_summary(subs, zones, areas, dists)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(areas)} rough nearest-substation areas -> {SERVICE_AREAS_PATH}")
    print(f"{len(dists)} suburbs -> {SUBURB_DISTANCES_CSV}")
    print(f"summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
