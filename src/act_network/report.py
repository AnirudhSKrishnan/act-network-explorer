"""Data-quality report for the OSM power network (CSV + Markdown + GeoPackage).

Run with::

    python -m act_network.report

Reads ``data/processed/act_power.gpkg`` (EPSG:7855), builds the connectivity
graph (:mod:`act_network.topology`) and writes

* ``outputs/quality_issues.csv``   -- one row per issue, with OSM id / link / lon-lat
* ``outputs/quality_report.md``    -- the human-readable report
* ``outputs/quality_summary.json`` -- machine-readable counts for later build steps
* ``data/processed/topology.gpkg`` -- layers ``quality_issues`` and ``components``

Every number in the outputs is computed here; nothing is hand-typed.

Issue groups (so a raw count is never mistaken for an error count):

* ``likely_error`` -- things that are wrong in the data as mapped: transmission
  dangling ends, transmission sections with no substation, voltage conflicts
  other than LV-vs-11/22 kV (e.g. 22 kV meeting 11 kV, or transmission meeting
  distribution, with no substation), duplicate or substantially overlapping
  lines, same-name duplicate substations.
* ``to_check`` -- unconnected LV / 11 kV / untagged ends, floating
  distribution sections, LV meeting 11/22 kV with no transformer mapped, and
  nested substation footprints. Radial feeders legitimately end at their last
  pole and OSM rarely maps the transformers that link LV to 11 kV, so many of
  these are normal network structure or simply unmapped equipment.
* ``tag_gap`` -- lines with no usable voltage tag.

Data: public OpenStreetMap data (c) OpenStreetMap contributors (ODbL) -- NOT
official network operator data; OSM power mapping is incomplete.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from act_network.clean import (
    BOUNDARY_LAYER,
    LV_MAX_V,
    TRANSMISSION_MIN_V,
    FOOTPRINTS_LAYER,
    LINES_LAYER,
    PROCESSED_DIR,
    PROCESSED_PATH,
    PROJECT_CRS,
    SUBSTATIONS_LAYER,
    TRANSFORMERS_LAYER,
)
from act_network.topology import (
    DEFAULT_TOL,
    Topology,
    build_graph,
    component_table,
    voltages_compatible,
)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
ISSUES_CSV = OUTPUTS_DIR / "quality_issues.csv"
REPORT_MD = OUTPUTS_DIR / "quality_report.md"
SUMMARY_JSON = OUTPUTS_DIR / "quality_summary.json"
TOPOLOGY_PATH = PROCESSED_DIR / "topology.gpkg"
ISSUES_LAYER = "quality_issues"
COMPONENTS_LAYER = "components"

NEAR_MISS_M = 20.0  # a dangling end this close to another line is a probable unjoined gap
OVERLAP_BUFFER_M = 1.0  # two lines "share" length where one lies within 1 m of the other
OVERLAP_MIN_M = 10.0  # ... and a substantial overlap shares at least 10 m
OVERLAP_MIN_FRAC = 0.5  # ... which is at least half of the shorter line
FOOTPRINT_NEST_FRAC = 0.5  # substation footprints overlapping by >= 50% of the smaller

DATA_NOTE = (
    "Public OpenStreetMap data, not official network operator data; OSM power data is "
    "incomplete. © OpenStreetMap contributors, ODbL."
)

ISSUE_COLUMNS = [
    "issue_type", "group", "subtype", "osm_id", "element", "line_class", "voltage",
    "details", "related_osm_ids", "component_id", "lon", "lat", "osm_url",
]

# ---------------------------------------------------------------------------
# Checks (pure functions)
# ---------------------------------------------------------------------------


def find_dangling_ends(
    topo: Topology, boundary=None, near_miss_m: float = NEAR_MISS_M
) -> pd.DataFrame:
    """Original way endpoints that touch no other line and no substation/transformer.

    An endpoint is dangling when its junction node has (a) no substation or
    transformer link and (b) no line edge belonging to a *different* way (so a
    T-junction split, an end-to-end join or a shared node all count as
    connected). Closed rings are never dangling. Split points are never
    candidates -- only the two original ends of each way.

    Extra columns: ``inside_study_area`` (endpoint inside ``boundary``; ends
    outside the ACT are where the download stops, not data errors),
    ``nearest_line_idx`` / ``nearest_m`` (closest other way within
    ``near_miss_m``) and ``near_miss``.
    """
    lines = topo.lines
    closed = shapely.is_closed(lines.geometry.values)
    ep = topo.endpoints
    mask = [
        (not links) and (not closed[li]) and topo.line_degree_others(n, li) == 0
        for li, n, links in zip(ep["line_idx"], ep["node"], ep["device_links"])
    ]
    out = ep[np.array(mask, dtype=bool)].copy()
    pts = shapely.points(out["x"].to_numpy(), out["y"].to_numpy())
    out["inside_study_area"] = (
        shapely.within(pts, boundary) if boundary is not None else np.ones(len(out), bool)
    )
    nearest_idx = np.full(len(out), -1)
    nearest_d = np.full(len(out), np.nan)
    if len(out):
        geoms = lines.geometry.values
        tree = shapely.STRtree(geoms)
        p_i, l_i = tree.query(pts, predicate="dwithin", distance=near_miss_m)
        own = out["line_idx"].to_numpy()[p_i] == l_i
        p_i, l_i = p_i[~own], l_i[~own]
        d = shapely.distance(pts[p_i], geoms[l_i])
        for p, li, dd in zip(p_i, l_i, d):
            if np.isnan(nearest_d[p]) or dd < nearest_d[p]:
                nearest_d[p], nearest_idx[p] = dd, li
    out["nearest_line_idx"] = nearest_idx
    out["nearest_m"] = nearest_d
    out["near_miss"] = nearest_idx >= 0
    return out.reset_index(drop=True)


def classify_components(comps: pd.DataFrame, topo: Topology, boundary=None) -> pd.DataFrame:
    """Add ``tier`` and ``crosses_boundary`` to :func:`topology.component_table` output.

    tier:
    * ``fed`` -- contains a substation node
    * ``crosses_boundary`` -- no substation, but part of it lies outside the ACT
      (it may be fed from NSW; not counted as an issue)
    * ``transformer_only`` -- no substation but at least one transformer: the
      11 kV side upstream of the transformer is not mapped / not connected
    * ``isolated`` -- neither a substation nor a transformer
    """
    comps = comps.copy()
    geoms = topo.lines.geometry.values
    if boundary is not None:
        inside = shapely.within(geoms, boundary)
        crosses = [not bool(inside[idx].all()) for idx in comps["line_idx"]]
    else:
        crosses = [False] * len(comps)
    comps["crosses_boundary"] = crosses
    tiers = []
    for has_sub, cr, has_tx in zip(comps["has_substation"], crosses, comps["has_transformer"]):
        if has_sub:
            tiers.append("fed")
        elif cr:
            tiers.append("crosses_boundary")
        elif has_tx:
            tiers.append("transformer_only")
        else:
            tiers.append("isolated")
    comps["tier"] = tiers
    return comps


def find_voltage_conflicts(topo: Topology) -> pd.DataFrame:
    """Junctions where ways with incompatible voltages meet with no substation/transformer.

    Definition: at a junction node with no substation/transformer link, two
    different ways whose *known* voltage sets are disjoint (e.g. ``11000`` and
    ``415``, or ``132000`` and ``11000``). Multi-valued tags count as all their
    voltages, so ``"11000;415"`` meeting ``"415"`` is fine (shared-pole LV), and
    a way with no voltage tag never conflicts (it is a ``missing_voltage`` issue
    instead). One row per junction node.
    """
    G = topo.graph
    vsets = topo.lines["vset"].to_numpy()
    rows = []
    for n, data in G.nodes(data=True):
        if data.get("kind") != "junction" or topo.node_has_device(n):
            continue
        idx = sorted({d["line_idx"] for _, _, d in G.edges(n, data=True) if d["kind"] == "line"})
        if len(idx) < 2:
            continue
        bad = [
            (a, b) for i, a in enumerate(idx) for b in idx[i + 1:]
            if not voltages_compatible(vsets[a], vsets[b])
        ]
        if bad:
            involved = sorted({i for pair in bad for i in pair})
            rows.append({"node": n, "x": data["x"], "y": data["y"],
                         "line_idx": involved, "n_conflicting_pairs": len(bad)})
    return pd.DataFrame(rows, columns=["node", "x", "y", "line_idx", "n_conflicting_pairs"])


def conflict_kind(a: frozenset[int], b: frozenset[int]) -> str:
    """Classify an incompatible voltage pair (both known, disjoint).

    * ``lv_vs_hv_distribution`` -- LV (<= 1 kV) meeting 11/22 kV with no transformer
      mapped: almost always an unmapped pole-top / kiosk transformer, i.e. missing
      equipment rather than a wrong line (``to_check``).
    * ``transmission_vs_distribution`` -- >= 66 kV meeting < 66 kV directly.
    * ``distribution_vs_distribution`` -- e.g. 22 kV meeting 11 kV directly.
    * ``other`` -- anything else (e.g. 132 kV meeting 66 kV).
    """
    hi, lo = (a, b) if max(a) >= max(b) else (b, a)
    if max(lo) <= LV_MAX_V and max(hi) > LV_MAX_V and max(hi) < TRANSMISSION_MIN_V:
        return "lv_vs_hv_distribution"
    if max(hi) >= TRANSMISSION_MIN_V and max(lo) < TRANSMISSION_MIN_V:
        return "transmission_vs_distribution"
    if max(hi) < TRANSMISSION_MIN_V:
        return "distribution_vs_distribution"
    return "other"


_CONFLICT_SEVERITY = ["lv_vs_hv_distribution", "other", "distribution_vs_distribution",
                      "transmission_vs_distribution"]


def find_missing_voltage(lines: gpd.GeoDataFrame) -> pd.Series:
    """Boolean mask: lines with no parseable voltage tag."""
    if "voltage_missing" in lines:
        return lines["voltage_missing"].astype(bool)
    return lines["voltages"].fillna("").eq("")


def find_duplicate_lines(lines: gpd.GeoDataFrame) -> pd.DataFrame:
    """Exact duplicate geometries (same vertices, either direction).

    Returns one row per *redundant copy* (``line_idx``) with ``original_idx``
    (lowest osm_id in its group) -- i.e. the number of ways someone would delete.
    """
    norm = shapely.normalize(lines.geometry.values)
    key = pd.Series(shapely.to_wkb(norm), index=lines.index)
    order = lines.assign(_k=key).reset_index().sort_values(["_k", "osm_id"])
    rows = []
    for _, grp in order.groupby("_k", sort=False):
        if len(grp) > 1:
            first = int(grp["index"].iloc[0])
            rows += [{"line_idx": int(i), "original_idx": first} for i in grp["index"].iloc[1:]]
    return pd.DataFrame(rows, columns=["line_idx", "original_idx"])


def find_overlapping_lines(
    lines: gpd.GeoDataFrame,
    buffer_m: float = OVERLAP_BUFFER_M,
    min_m: float = OVERLAP_MIN_M,
    min_frac: float = OVERLAP_MIN_FRAC,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Pairs of ways where one lies on top of the other for a substantial length.

    Shared length = length of the shorter way lying within ``buffer_m`` of the
    longer one. A pair is a *substantial* overlap when the shared length is at
    least ``min_m`` metres AND at least ``min_frac`` of the shorter way.
    Partial overlaps below the fraction (typically separately mapped circuits
    converging onto the same towers near a substation) are only counted
    (second return value), not reported as issues. Exact duplicates are
    excluded (``exclude_pairs``).
    """
    lines = lines.reset_index(drop=True)
    geoms = lines.geometry.values
    lengths = shapely.length(geoms)
    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate="dwithin", distance=buffer_m)
    m = a < b
    a, b = a[m], b[m]
    # shorter way first
    swap = lengths[a] > lengths[b]
    short = np.where(swap, b, a)
    long_ = np.where(swap, a, b)
    shared_geom = shapely.intersection(geoms[short], shapely.buffer(geoms[long_], buffer_m, cap_style="flat"))
    shared = shapely.length(shared_geom)
    frac = shared / np.maximum(lengths[short], 1e-9)
    sizable = shared >= min_m
    substantial = sizable & (frac >= min_frac)
    exclude = exclude_pairs or set()
    rows = []
    for s, lg, sh, fr, g in zip(short[substantial], long_[substantial], shared[substantial],
                                frac[substantial], shared_geom[substantial]):
        if (min(s, lg), max(s, lg)) in exclude:
            continue
        rows.append({"line_idx": int(s), "other_idx": int(lg), "shared_m": float(sh),
                     "frac_of_shorter": float(fr), "geometry": g})
    n_partial = int((sizable & ~substantial).sum())
    return pd.DataFrame(rows, columns=["line_idx", "other_idx", "shared_m", "frac_of_shorter", "geometry"]), n_partial


def find_duplicate_substations(
    substations: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame | None = None
) -> pd.DataFrame:
    """Substations mapped more than once.

    * ``same_name``: same (case/space-normalised) name -> one row per extra copy.
    * ``nested_footprint``: two footprints overlapping by >= 50% of the smaller
      one (may be a genuine site-within-a-site, so it is a "to check").
    * ``point_in_footprint``: a point substation inside another's footprint.
    """
    rows = []
    subs = substations.reset_index(drop=True)
    names = subs["name"].fillna("").astype(str).str.lower().str.split().str.join(" ")
    for nm, grp in subs[names != ""].groupby(names[names != ""]):
        if len(grp) > 1:
            grp = grp.sort_values("osm_id")
            first = grp.iloc[0]
            for _, r in grp.iloc[1:].iterrows():
                rows.append({"osm_id": int(r["osm_id"]), "other_osm_id": int(first["osm_id"]),
                             "subtype": "same_name", "name": r["name"]})
    if footprints is not None and len(footprints):
        fp = footprints.reset_index(drop=True)
        fg = fp.geometry.values
        tree = shapely.STRtree(fg)
        a, b = tree.query(fg, predicate="intersects")
        for i, j in zip(a, b):
            if i >= j:
                continue
            inter = shapely.area(shapely.intersection(fg[i], fg[j]))
            smaller = min(fg[i].area, fg[j].area)
            if smaller > 0 and inter / smaller >= FOOTPRINT_NEST_FRAC:
                small, big = (i, j) if fg[i].area <= fg[j].area else (j, i)
                rows.append({"osm_id": int(fp["osm_id"].iat[small]),
                             "other_osm_id": int(fp["osm_id"].iat[big]),
                             "subtype": "nested_footprint", "name": fp["name"].iat[small]})
        fp_ids = set(int(x) for x in fp["osm_id"])
        pts = subs[~subs["osm_id"].astype(int).isin(fp_ids)]
        if len(pts):
            p_i, f_i = tree.query(pts.geometry.values, predicate="within")
            for p, f in zip(p_i, f_i):
                rows.append({"osm_id": int(pts["osm_id"].iat[p]),
                             "other_osm_id": int(fp["osm_id"].iat[f]),
                             "subtype": "point_in_footprint", "name": pts["name"].iat[p]})
    df = pd.DataFrame(rows, columns=["osm_id", "other_osm_id", "subtype", "name"])
    return df.drop_duplicates(["osm_id", "other_osm_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assemble issues
# ---------------------------------------------------------------------------


def _vstr(v) -> str | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else str(v)


def _ids(lines: gpd.GeoDataFrame, idx) -> str:
    return ";".join(str(int(lines["osm_id"].iat[i])) for i in idx)


def _kv(vset) -> str:
    if not vset:
        return "no voltage"
    return "/".join(f"{v / 1000:g} kV" if v >= 1000 else f"{v} V" for v in sorted(vset, reverse=True))


def _truncate_ids(ids: str | None, keep: int = 25) -> str | None:
    if not isinstance(ids, str):
        return ids
    parts = ids.split(";")
    if len(parts) <= keep:
        return ids
    return ";".join(parts[:keep]) + f";...(+{len(parts) - keep} more; see topology.gpkg components layer)"


def build_issues(
    topo: Topology,
    substations: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame | None,
    boundary=None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Run every check; return (issues GeoDataFrame in EPSG:7855, intermediate results)."""
    lines = topo.lines
    crs = lines.crs or PROJECT_CRS
    geoms = lines.geometry.values
    rows: list[dict] = []

    def line_row(issue_type, group, subtype, li, details, geom, related=None, comp=None):
        rows.append({
            "issue_type": issue_type, "group": group, "subtype": subtype,
            "osm_id": int(lines["osm_id"].iat[li]),
            "element": lines["element"].iat[li] if "element" in lines else "way",
            "line_class": lines["line_class"].iat[li] if "line_class" in lines else None,
            "voltage": _vstr(lines["voltages"].iat[li]),
            "details": details,
            "related_osm_ids": related if related is not None else str(int(lines["osm_id"].iat[li])),
            "component_id": comp, "geometry": geom,
        })

    comps = classify_components(component_table(topo), topo, boundary)
    if "sub_class" in substations:
        grid_ids = set(substations.loc[substations["sub_class"].isin(["zone", "transmission"]), "osm_id"].astype(int))
    else:
        grid_ids = set()
    comps["has_grid_substation"] = [bool(grid_ids & set(map(int, s))) for s in comps["substations"]]
    comp_of_line = np.full(len(lines), -1)
    for cid, idx in zip(comps["component_id"], comps["line_idx"]):
        comp_of_line[idx] = cid

    # 1. floating components
    sub_pts = substations.geometry.values if len(substations) else None
    sub_tree = shapely.STRtree(sub_pts) if sub_pts is not None else None
    for c in comps[comps["tier"].isin(["isolated", "transformer_only"])].itertuples():
        idx = list(c.line_idx)
        longest = max(idx, key=lambda i: geoms[i].length)
        group = "likely_error" if c.top_class == "transmission" else "to_check"
        tx = f"; {len(c.transformers)} transformer(s) but no substation" if c.has_transformer else ""
        near = ""
        if sub_tree is not None:
            union = shapely.multilinestrings(list(geoms[idx]))
            j = int(sub_tree.nearest(union))
            nm = substations["name"].iat[j] if "name" in substations else None
            near = (f"; nearest substation: {nm if isinstance(nm, str) else 'unnamed ' + str(substations['osm_id'].iat[j])}"
                    f" {shapely.distance(union, sub_pts[j]):.0f} m away")
        details = (f"{c.n_lines} way(s), {c.length_m / 1000:.2f} km, voltages: "
                   f"{_kv(set(c.voltages))}{tx}; not connected to any mapped substation{near}")
        line_row("floating_component", group, f"{c.tier}/{c.top_class}", longest, details,
                 shapely.multilinestrings(list(geoms[idx])), _ids(lines, idx), int(c.component_id))

    # 2. dangling ends
    dang = find_dangling_ends(topo, boundary)
    sub_shapes = {int(i): g for i, g in zip(substations["osm_id"], substations.geometry)}
    if footprints is not None:
        sub_shapes.update({int(i): g for i, g in zip(footprints["osm_id"], footprints.geometry)})
    sub_names = {int(i): (n if isinstance(n, str) else f"unnamed {int(i)}")
                 for i, n in zip(substations["osm_id"], substations.get("name", pd.Series(None, index=substations.index)))}
    shape_ids = list(sub_shapes)
    shape_tree = shapely.STRtree(list(sub_shapes.values())) if shape_ids else None

    def near_sub(x, y, within=100.0) -> str:
        if shape_tree is None:
            return ""
        p = shapely.Point(x, y)
        j = int(shape_tree.nearest(p))
        dist = shapely.distance(p, shape_tree.geometries[j])
        if dist > within:
            return ""
        sid = shape_ids[j]
        return f"; {dist:.0f} m from substation '{sub_names.get(sid, sid)}' (outside the snapping tolerance)"
    for d in dang[dang["inside_study_area"]].itertuples():
        li = int(d.line_idx)
        cls = lines["line_class"].iat[li] if "line_class" in lines else "unknown"
        group = "likely_error" if cls == "transmission" else "to_check"
        if d.near_miss:
            other = int(d.nearest_line_idx)
            ocls = lines["line_class"].iat[other] if "line_class" in lines else "unknown"
            head = (f"{d.end} of way ends {d.nearest_m:.1f} m from way "
                    f"{int(lines['osm_id'].iat[other])} ({_kv(lines['vset'].iat[other])})")
            if {cls, ocls} == {"transmission", "distribution"}:
                details = (f"{head}; not joined because transmission and distribution only connect "
                           f"through a substation: unmapped substation, or an underbuilt/crossing circuit")
            else:
                details = f"{head} without joining it: probable unjoined gap"
            sub = "near_miss"
            related = _ids(lines, [li, other])
        else:
            details = (f"{d.end} of way touches no other line, substation or transformer "
                       f"(nothing within {NEAR_MISS_M:g} m): end of feeder or unmapped connection")
            sub = "isolated_end"
            related = None
        details += near_sub(d.x, d.y)
        line_row("dangling_end", group, f"{cls}/{sub}", li, details,
                 shapely.Point(d.x, d.y), related, int(comp_of_line[li]))

    # 3. missing voltage
    for li in np.flatnonzero(find_missing_voltage(lines).to_numpy()):
        power = lines["power"].iat[li] if "power" in lines else "line"
        line_row("missing_voltage", "tag_gap", str(power), int(li),
                 f"power={power} way with no usable voltage tag ({geoms[li].length:.0f} m)",
                 geoms[li], comp=int(comp_of_line[li]))

    # 4. voltage conflicts
    conf = find_voltage_conflicts(topo)
    vsets = lines["vset"].to_numpy()
    for c in conf.itertuples():
        idx = list(c.line_idx)
        kinds = {conflict_kind(vsets[a], vsets[b]) for k, a in enumerate(idx) for b in idx[k + 1:]
                 if not voltages_compatible(vsets[a], vsets[b])}
        kind = max(kinds, key=_CONFLICT_SEVERITY.index)
        lowest = min(idx, key=lambda i: max(vsets[i]))
        parts = ", ".join(f"{int(lines['osm_id'].iat[i])} ({_kv(vsets[i])})" for i in idx)
        if kind == "lv_vs_hv_distribution":
            group = "to_check"
            why = "Almost always an unmapped pole-top or kiosk transformer (missing equipment)"
        else:
            group = "likely_error"
            why = "A substation/transformer is missing here or a voltage tag is wrong"
        details = (f"ways with different voltages meet here with no substation or transformer "
                   f"mapped: {parts}. {why}")
        line_row("voltage_conflict", group, kind, lowest, details,
                 shapely.Point(c.x, c.y), _ids(lines, idx), int(comp_of_line[lowest]))

    # 5. duplicates / overlaps
    dups = find_duplicate_lines(lines)
    for d in dups.itertuples():
        li, oi = int(d.line_idx), int(d.original_idx)
        line_row("duplicate_line", "likely_error", "exact_geometry", li,
                 f"identical geometry to way {int(lines['osm_id'].iat[oi])}", geoms[li],
                 _ids(lines, [li, oi]), int(comp_of_line[li]))
    dup_pairs = {(min(int(a), int(b)), max(int(a), int(b)))
                 for a, b in zip(dups["line_idx"], dups["original_idx"])}
    over, n_partial = find_overlapping_lines(lines, exclude_pairs=dup_pairs)
    # overlaps inside a substation (line=bay/busbar ways, or lying within a footprint)
    # get their own subtype prefix so they are not mistaken for open-network errors
    internal_flag = (lines["is_substation_internal"].fillna(False).astype(bool).to_numpy()
                     if "is_substation_internal" in lines else np.zeros(len(lines), bool))
    fp_geoms = (list(footprints.geometry.buffer(OVERLAP_BUFFER_M + 5.0))
                if footprints is not None and len(footprints) else [])
    fp_tree = shapely.STRtree(fp_geoms) if fp_geoms else None
    over_site = []
    for o in over.itertuples():
        li, oi = int(o.line_idx), int(o.other_idx)
        site = None
        if fp_tree is not None:
            hits = fp_tree.query(o.geometry, predicate="covered_by")
            if len(hits):
                site = int(footprints["osm_id"].iat[int(hits[0])])
        if site is None and internal_flag[li] and internal_flag[oi]:
            site = -1  # bay/busbar pair outside any mapped footprint
        over_site.append(site)
        va, vb = lines["vset"].iat[li], lines["vset"].iat[oi]
        sub = ("unknown_voltage" if not va or not vb
               else "same_voltage" if va == vb else "different_voltage")
        details = (f"{o.shared_m:.0f} m ({o.frac_of_shorter:.0%} of this way) lies within "
                   f"{OVERLAP_BUFFER_M:g} m of way {int(lines['osm_id'].iat[oi])} "
                   f"({_kv(va)} vs {_kv(vb)})")
        if site is not None:
            sub = f"inside_substation/{sub}"
            details += ("; inside a substation" + (f" (footprint way {site})" if site > 0 else "")
                        + ": bay/busbar conductors, possibly mapped twice")
        line_row("overlapping_lines", "likely_error", sub, li, details, o.geometry,
                 _ids(lines, [li, oi]), int(comp_of_line[li]))

    # 6. duplicate substations
    dsubs = find_duplicate_substations(substations, footprints)
    sub_geom = {int(i): g for i, g in zip(substations["osm_id"], substations.geometry)}
    sub_elem = dict(zip(substations["osm_id"].astype(int),
                        substations["element"] if "element" in substations else ["way"] * len(substations)))
    if footprints is not None:
        sub_geom.update({int(i): g for i, g in zip(footprints["osm_id"], footprints.geometry)})
    for d in dsubs.itertuples():
        group = "likely_error" if d.subtype == "same_name" else "to_check"
        what = {"same_name": "has the same name as",
                "nested_footprint": "footprint lies mostly inside the footprint of",
                "point_in_footprint": "point lies inside the footprint of"}[d.subtype]
        rows.append({
            "issue_type": "duplicate_substation", "group": group, "subtype": d.subtype,
            "osm_id": int(d.osm_id), "element": sub_elem.get(int(d.osm_id), "way"),
            "line_class": None, "voltage": None,
            "details": f"substation '{d.name}' {what} substation {int(d.other_osm_id)}",
            "related_osm_ids": f"{int(d.osm_id)};{int(d.other_osm_id)}", "component_id": None,
            "geometry": sub_geom.get(int(d.osm_id)),
        })

    issues = gpd.GeoDataFrame(rows, columns=[c for c in ISSUE_COLUMNS if c not in ("lon", "lat", "osm_url")] + ["geometry"],
                              geometry="geometry", crs=crs)
    if len(issues):
        rp = issues.geometry.representative_point().to_crs(4326)
        issues["lon"], issues["lat"] = rp.x.round(6), rp.y.round(6)
    else:
        issues["lon"], issues["lat"] = [], []
    issues["osm_url"] = [f"https://www.openstreetmap.org/{e}/{i}" for e, i in zip(issues["element"], issues["osm_id"])]
    issues["component_id"] = issues["component_id"].astype("Int64")
    issues = issues[ISSUE_COLUMNS + ["geometry"]]
    over = over.assign(substation_site=pd.array(over_site, dtype="Int64"))
    extra = {"components": comps, "dangling": dang, "conflicts": conf, "duplicates": dups,
             "overlaps": over, "n_partial_overlaps": n_partial, "dup_substations": dsubs}
    return issues, extra


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------


def _count(df: pd.DataFrame, by: list[str]) -> dict:
    if df.empty:
        return {}
    s = df.groupby(by).size()
    return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v) for k, v in s.items()}


def summarise(topo: Topology, issues: gpd.GeoDataFrame, extra: dict, *, sensitivity=None,
              no_tsnap_dangling=None, fed_share_footprint_only=None, runtime_s=None) -> dict:
    lines = topo.lines
    comps = extra["components"]
    dang = extra["dangling"]
    total_len = float(lines.geometry.length.sum())
    cls_series = lines.get("line_class", pd.Series("unknown", index=lines.index))
    km_by_class = (lines.geometry.length.groupby(cls_series).sum() / 1000).round(1).to_dict()
    in_act_len = (lines["length_in_act_m"].astype(float) if "length_in_act_m" in lines else None)
    internal = (lines["is_substation_internal"].fillna(False).astype(bool)
                if "is_substation_internal" in lines else pd.Series(False, index=lines.index))
    largest = comps.iloc[0] if len(comps) else None
    inside = dang[dang["inside_study_area"]]
    cls_of = lines["line_class"].to_numpy() if "line_class" in lines else np.full(len(lines), "unknown")
    inside_cls = pd.Series(cls_of[inside["line_idx"].to_numpy()], dtype=object)
    near = inside["near_miss"].to_numpy()

    def ways(sub: pd.DataFrame) -> int:
        ids = set()
        for r in sub["related_osm_ids"].dropna():
            ids.update(r.split(";"))
        return len(ids)

    by_group = {g: int((issues["group"] == g).sum()) for g in ("likely_error", "to_check", "tag_gap")}
    floating = comps[comps["tier"].isin(["isolated", "transformer_only"])]
    summary = {
        "data_note": DATA_NOTE,
        "params": {**topo.params, "near_miss_m": NEAR_MISS_M, "overlap_buffer_m": OVERLAP_BUFFER_M,
                   "overlap_min_m": OVERLAP_MIN_M, "overlap_min_frac": OVERLAP_MIN_FRAC,
                   "footprint_nest_frac": FOOTPRINT_NEST_FRAC},
        "network": {
            "n_lines": int(len(lines)),
            "total_km": round(total_len / 1000, 1),
            "km_by_line_class": km_by_class,
            "total_km_in_act": round(float(in_act_len.sum()) / 1000, 1) if in_act_len is not None else None,
            "km_in_act_by_line_class": ((in_act_len.groupby(cls_series).sum() / 1000).round(1).to_dict()
                                        if in_act_len is not None else None),
            "km_substation_bays_busbars": round(float(lines.geometry.length[internal.to_numpy()].sum()) / 1000, 2),
            "n_substation_bay_busbar_ways": int(internal.sum()),
            "n_substations": topo.stats["n_substation_nodes"],
            "n_transformers": topo.stats["n_transformer_nodes"],
            "n_transformers_by_source": extra.get("transformers_by_source") or {},
            "graph_nodes": topo.graph.number_of_nodes(),
            "graph_edges": topo.graph.number_of_edges(),
            "n_components": int(len(comps)),
            "n_components_with_substation": int(comps["has_substation"].sum()),
            "largest_component_share_of_length": round(float(largest["length_m"]) / total_len, 3) if largest is not None else None,
            "largest_component_share_of_lines": round(int(largest["n_lines"]) / len(lines), 3) if largest is not None else None,
            "largest_component_km": round(float(largest["length_m"]) / 1000, 1) if largest is not None else None,
            "share_of_length_in_fed_components": round(float(comps.loc[comps["has_substation"], "length_m"].sum()) / total_len, 3) if total_len else None,
            "share_of_length_with_zone_or_transmission_substation": round(float(comps.loc[comps["has_grid_substation"], "length_m"].sum()) / total_len, 3) if total_len and "has_grid_substation" in comps else None,
            "share_of_length_fed_footprint_only_rule": fed_share_footprint_only,
            "components_by_tier": _count(comps, ["tier"]),
            "km_by_tier": {k: round(float(v) / 1000, 1) for k, v in comps.groupby("tier")["length_m"].sum().items()},
        },
        "snapping": {**topo.stats,
                     "blocked_tsnaps_by_class": _count(
                         pd.DataFrame({"s": cls_of[topo.blocked_snaps["line_idx"].to_numpy(dtype=int)],
                                       "t": cls_of[topo.blocked_snaps["target_idx"].to_numpy(dtype=int)]}), ["s", "t"]),
                     "tjunctions_by_class": _count(
                         pd.DataFrame({"s": cls_of[topo.tjunctions["line_idx"].to_numpy(dtype=int)],
                                       "t": cls_of[topo.tjunctions["target_idx"].to_numpy(dtype=int)]}), ["s", "t"])},
        "checks": {
            "floating_components": {
                "count": int(len(floating)),
                "km": round(float(floating["length_m"].sum()) / 1000, 1),
                "ways": int(floating["n_lines"].sum()),
                "by_tier_and_top_class": _count(floating, ["tier", "top_class"]),
                "by_size": _size_buckets(floating),
                "excluded_crossing_act_boundary": int((comps["tier"] == "crosses_boundary").sum()),
            },
            "dangling_ends": {
                "count": int(len(inside)),
                "by_line_class": {k: int(v) for k, v in inside_cls.value_counts().items()},
                "near_miss_by_line_class": {k: int(v) for k, v in inside_cls[near].value_counts().items()},
                "isolated_by_line_class": {k: int(v) for k, v in inside_cls[~near].value_counts().items()},
                "excluded_outside_act": int((~dang["inside_study_area"]).sum()),
                "count_without_tjunction_snapping": no_tsnap_dangling,
            },
            "missing_voltage": {
                "count": int((issues["issue_type"] == "missing_voltage").sum()),
                "km": round(float(lines.geometry.length[find_missing_voltage(lines).to_numpy()].sum()) / 1000, 1),
                "by_power_tag": _count(issues[issues["issue_type"] == "missing_voltage"], ["subtype"]),
            },
            "voltage_conflicts": {
                "junctions": int(len(extra["conflicts"])),
                "by_kind": _count(issues[issues["issue_type"] == "voltage_conflict"], ["subtype"]),
                "components_affected": int(issues.loc[issues["issue_type"] == "voltage_conflict", "component_id"].nunique()),
                "by_voltage_pair": _conflict_pairs(topo, extra["conflicts"]),
            },
            "duplicate_lines": int(len(extra["duplicates"])),
            "overlapping_lines": {
                "count": int(len(extra["overlaps"])),
                "inside_substations": int(extra["overlaps"]["substation_site"].notna().sum())
                if "substation_site" in extra["overlaps"] else 0,
                "substation_sites": int(extra["overlaps"]["substation_site"].dropna().nunique())
                if "substation_site" in extra["overlaps"] else 0,
                "by_subtype": _count(issues[issues["issue_type"] == "overlapping_lines"], ["subtype"]),
                "partial_overlaps_not_reported": extra["n_partial_overlaps"],
            },
            "duplicate_substations": _count(extra["dup_substations"], ["subtype"]),
        },
        "issues": {
            "total_rows": int(len(issues)),
            "by_group": by_group,
            "likely_error_outside_substations": int(
                ((issues["group"] == "likely_error")
                 & ~issues["subtype"].fillna("").str.startswith("inside_substation")).sum()),
            "by_type": _count(issues, ["issue_type"]),
            "by_type_and_group": _count(issues, ["issue_type", "group"]),
            "distinct_osm_objects_flagged": ways(issues),
            "distinct_osm_objects_flagged_by_group": {
                g: ways(issues[issues["group"] == g]) for g in ("likely_error", "to_check", "tag_gap")},
        },
        "sensitivity": sensitivity or [],
        "runtime_s": runtime_s,
    }
    return summary


def _size_buckets(comps: pd.DataFrame) -> dict:
    bins = [(1, 1, "1 way"), (2, 10, "2-10 ways"), (11, 100, "11-100 ways"), (101, 10**9, ">100 ways")]
    out = {}
    for lo, hi, label in bins:
        sel = comps[(comps["n_lines"] >= lo) & (comps["n_lines"] <= hi)]
        out[label] = {"components": int(len(sel)), "km": round(float(sel["length_m"].sum()) / 1000, 1)}
    return out


def _conflict_pairs(topo: Topology, conf: pd.DataFrame) -> dict:
    counts: dict[str, int] = {}
    vsets = topo.lines["vset"].to_numpy()
    for idx in conf["line_idx"]:
        pairs = set()
        for i, a in enumerate(idx):
            for b in idx[i + 1:]:
                if not voltages_compatible(vsets[a], vsets[b]):
                    pairs.add(" vs ".join(sorted([_kv(vsets[a]), _kv(vsets[b])], reverse=True)))
        for p in pairs:
            counts[p] = counts.get(p, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def sensitivity_runs(lines, substations, footprints, transformers, boundary, tols=(2.0, 5.0, 10.0)) -> list[dict]:
    """Re-run graph + headline checks at other snapping tolerances (reported for transparency)."""
    out = []
    for tol in tols:
        topo = build_graph(lines, substations, footprints, transformers, tol=tol)
        comps = classify_components(component_table(topo), topo, boundary)
        dang = find_dangling_ends(topo, boundary)
        inside = dang[dang["inside_study_area"]]
        cls = topo.lines["line_class"].to_numpy()[inside["line_idx"].to_numpy()]
        out.append({
            "tol_m": tol,
            "components": int(len(comps)),
            "floating_components": int(comps["tier"].isin(["isolated", "transformer_only"]).sum()),
            "dangling_ends": int(len(inside)),
            "dangling_transmission": int((cls == "transmission").sum()),
            "voltage_conflicts": int(len(find_voltage_conflicts(topo))),
            "fed_length_share": round(float(comps.loc[comps["has_substation"], "length_m"].sum())
                                      / float(topo.lines.geometry.length.sum()), 3),
        })
    return out


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _examples(issues: pd.DataFrame, n: int = 10) -> str:
    if issues.empty:
        return "_None found._"
    rows = [[f"[{r.element}/{r.osm_id}]({r.osm_url})", r.subtype, r.voltage or "", r.details,
             f"{r.lat:.5f}, {r.lon:.5f}"] for r in issues.head(n).itertuples()]
    more = f"\n\n_{len(issues) - n} more in `quality_issues.csv`._" if len(issues) > n else ""
    return _md_table(["OSM object", "subtype", "voltage", "details", "lat, lon"], rows) + more


def render_markdown(summary: dict, issues: pd.DataFrame) -> str:
    p, net, chk, iss = summary["params"], summary["network"], summary["checks"], summary["issues"]
    sn = summary["snapping"]
    d, f, mv, vc, ov = (chk["dangling_ends"], chk["floating_components"], chk["missing_voltage"],
                        chk["voltage_conflicts"], chk["overlapping_lines"])
    ds = chk["duplicate_substations"]

    def fmt(dct: dict) -> str:
        return ", ".join(f"{k}: {v}" for k, v in dct.items()) if dct else "0"

    by = issues.groupby("issue_type").size().to_dict()
    L = []
    L.append("# ACT power network: OpenStreetMap data-quality report\n")
    L.append(f"> **Data source:** {DATA_NOTE} Suburb/boundary data are not used in these checks "
             "except the ACT boundary (OSM). This report checks how the network is *mapped* in OSM; "
             "it says nothing about the condition of the real network.\n")
    L.append("Generated by `python -m act_network.report`. Every number below is computed by the code "
             "(also in `outputs/quality_summary.json`); the full issue list with OSM links is "
             "`outputs/quality_issues.csv` and the map layer is `data/processed/topology.gpkg`.\n")

    L.append("## Summary\n")
    L.append(_md_table(
        ["Group", "Issue rows", "Distinct OSM objects involved", "Meaning"],
        [["Likely errors", iss["by_group"]["likely_error"], iss["distinct_osm_objects_flagged_by_group"]["likely_error"],
          "probably wrong as mapped in OSM: transmission dead ends / floating transmission, voltage conflicts other than LV-vs-11/22 kV, duplicate or overlapping lines, duplicate substations"],
         ["To check", iss["by_group"]["to_check"], iss["distinct_osm_objects_flagged_by_group"]["to_check"],
          "unconnected LV / 11 kV / untagged ends, floating distribution sections and LV meeting 11/22 kV with no transformer mapped; many are normal feeder ends or unmapped equipment"],
         ["Tag gaps", iss["by_group"]["tag_gap"], iss["distinct_osm_objects_flagged_by_group"]["tag_gap"],
          "ways with no voltage tag"],
         ["**All**", f"**{iss['total_rows']}**", f"**{iss['distinct_osm_objects_flagged']}**", ""]]))
    L.append("\nOne OSM way can appear in several rows (e.g. an untagged isolated stub is one floating "
             "component, two dangling ends and one missing voltage). The distinct-object column counts every "
             "OSM object *involved* in an issue, including the partner way of an overlap or duplicate pair, so "
             "it is not a count of objects that are themselves wrong.\n")
    n_inside = ov.get("inside_substations", 0)
    if n_inside:
        L.append(f"Of the {iss['by_group']['likely_error']} likely-error rows, {n_inside} are overlapping "
                 f"conductors *inside* substations ({ov.get('substation_sites', 0)} site(s); mostly "
                 "`line=bay` / `line=busbar` ways that look mapped twice) and "
                 f"{iss['likely_error_outside_substations']} are elsewhere (not substation-internal).\n")

    L.append("### Counts per check\n")
    size_txt = {k: f"{v['components']} ({v['km']} km)" for k, v in f["by_size"].items()}
    rows = [
        ["1. Floating components (no substation)", f["count"],
         f"{f['km']} km, {f['ways']} ways; {fmt(f['by_tier_and_top_class'])}; by size: "
         f"{fmt(size_txt)}; "
         f"{f['excluded_crossing_act_boundary']} more cross the ACT boundary (excluded)"],
        ["2. Dangling ends", d["count"],
         f"by class: {fmt(d['by_line_class'])}; near-miss (other line < {p['near_miss_m']:g} m): "
         f"{fmt(d['near_miss_by_line_class'])}; {d['excluded_outside_act']} ends outside the ACT excluded"],
        ["3. Missing voltage", mv["count"], f"{mv['km']} km; by power tag: {fmt(mv['by_power_tag'])}"],
        ["4. Voltage conflicts (junctions)", vc["junctions"],
         f"in {vc['components_affected']} components; {fmt(vc['by_voltage_pair'])}; by kind: {fmt(vc['by_kind'])}"],
        ["5a. Exact duplicate lines", chk["duplicate_lines"], "redundant copies"],
        ["5b. Overlapping lines", ov["count"],
         f"{fmt(ov['by_subtype'])} ({ov.get('inside_substations', 0)} inside substations at "
         f"{ov.get('substation_sites', 0)} site(s)); {ov['partial_overlaps_not_reported']} partial overlaps "
         "below the threshold not reported"],
        ["5c. Duplicate substations", sum(ds.values()) if ds else 0, fmt(ds)],
    ]
    L.append(_md_table(["Check", "Count", "Breakdown"], rows))

    L.append("\n## Network size\n")
    L.append(_md_table(["Measure", "Value"], [
        ["Line/cable ways", net["n_lines"]],
        ["Total length of mapped ways (whole ways, incl. parts outside the ACT)",
         f"{net['total_km']} km ({fmt({k: f'{v} km' for k, v in net['km_by_line_class'].items()})})"],
        *([["Length inside the ACT boundary",
            f"{net['total_km_in_act']} km ({fmt({k: f'{v} km' for k, v in net['km_in_act_by_line_class'].items()})})"]]
          if net.get("total_km_in_act") is not None else []),
        ["of which substation bays/busbars (`line=bay`/`busbar`)",
         f"{net['km_substation_bays_busbars']} km ({net['n_substation_bay_busbar_ways']} ways)"],
        ["OSM substation features / transformers (graph nodes)",
         f"{net['n_substations']} / {net['n_transformers']}"
         + (f" (transformers by source: {fmt(net['n_transformers_by_source'])})"
            if net.get("n_transformers_by_source") else "")
         + "; substation features are not distinct sites (see duplicate substations)"],
        ["Graph nodes / edges", f"{net['graph_nodes']} / {net['graph_edges']}"],
        ["Connected components (with at least one line)", net["n_components"]],
        ["Components containing a substation", net["n_components_with_substation"]],
        ["Largest component", f"{net['largest_component_km']} km = {net['largest_component_share_of_length']:.1%} of line length, "
                              f"{net['largest_component_share_of_lines']:.1%} of ways"],
        ["Share of line length connected to any substation", f"{net['share_of_length_in_fed_components']:.1%}"
         + (f" ({net['share_of_length_fed_footprint_only_rule']:.1%} if polygon substations count only via their footprint, "
            "without the 30 m-from-centroid clause)" if net.get("share_of_length_fed_footprint_only_rule") is not None else "")],
        ["Share of line length connected to a zone or transmission substation",
         f"{net['share_of_length_with_zone_or_transmission_substation']:.1%}"],
        ["Components by tier", fmt(net["components_by_tier"])],
        ["Length by tier", fmt({k: f"{v} km" for k, v in net["km_by_tier"].items()})],
    ]))

    L.append("\n## Parameters\n")
    L.append(_md_table(["Parameter", "Value", "Used for"], [
        ["`tol`", f"{p['tol_m']:g} m", "snap line endpoints to each other and onto other lines' interiors (T-junctions)"],
        ["`footprint_tol`", f"{p['footprint_tol_m']:g} m", "endpoint inside a substation footprint buffered by this = connected"],
        ["`sub_tol`", f"{p['sub_tol_m']:g} m", "endpoint within this of a substation's point (centroid for polygon substations) = connected"],
        ["`tx_tol`", f"{p['tx_tol_m']:g} m", "endpoint within this of a transformer = connected"],
        ["`block_cross_class_tsnap`", p["block_cross_class_tsnap"], "never T-snap a transmission line onto a distribution line or vice versa"],
        ["`near_miss`", f"{p['near_miss_m']:g} m", "dangling end this close to another line = probable gap"],
        ["overlap", f">= {p['overlap_min_m']:g} m and >= {p['overlap_min_frac']:.0%} of the shorter way within {p['overlap_buffer_m']:g} m", "substantial overlap"],
        ["CRS", "EPSG:7855 (GDA2020 / MGA 55)", "all distances"],
    ]))

    L.append("\n### Snapping diagnostics\n")
    L.append(f"- {sn['n_endpoints']} way endpoints collapsed into {sn['n_junction_nodes']} junction nodes "
             f"(largest single-linkage cluster spans {sn['max_cluster_extent_m']} m).")
    L.append(f"- {sn['n_tjunction_snaps']} T-junction snaps (endpoint onto another way's interior): {fmt(sn['tjunctions_by_class'])}.")
    L.append(f"- Without T-junction snapping there would be {d['count_without_tjunction_snapping']} dangling ends inside the ACT "
             f"instead of {d['count']}: endpoint-only snapping would report every pole T-off as a dead end.")
    L.append(f"- {sn['n_blocked_cross_class_tsnaps']} T-snaps blocked because they would join transmission and "
             f"distribution directly ({fmt(sn['blocked_tsnaps_by_class'])}); these ends are reported as near-misses instead.")
    L.append(f"- {sn['n_shared_interior_nodes_joined']} OSM nodes shared by two or more ways mid-way (X-junctions) "
             f"were joined; {sn['n_shared_interior_nodes_not_joined']} were left unjoined because the ways have "
             "incompatible voltages (a shared pole carrying separate circuits).")
    L.append(f"- {sn['n_device_interior_attachments']} ways pass through a substation footprint or by a transformer "
             "without ending there; they are attached at that point.")
    if summary["sensitivity"]:
        L.append("\n**Sensitivity to `tol`** (other parameters unchanged):\n")
        L.append(_md_table(["tol (m)", "components", "floating", "dangling ends", "of which transmission",
                            "voltage conflicts", "length share fed"],
                           [[s["tol_m"], s["components"], s["floating_components"], s["dangling_ends"],
                             s["dangling_transmission"], s["voltage_conflicts"], f"{s['fed_length_share']:.1%}"]
                            for s in summary["sensitivity"]]))

    L.append("\n## How each check works, and its caveats\n")
    L.append("**Graph model.** Every OSM way is an edge (split where another way T-joins it). Endpoints within "
             f"`tol` are merged into one junction node. Substations and transformers are nodes of their own, linked "
             "to the endpoints that reach them. A component counts as *fed* only if it contains a substation; "
             "transformers connect LV to 11 kV but are not supply points.\n")
    L.append("**1. Floating components.** Connected pieces of network with no substation. `isolated` = no "
             "substation and no transformer; `transformer_only` = reaches a transformer but the 11 kV side behind "
             "it is not mapped or not joined. Pieces that cross the ACT boundary are excluded because they may be "
             "fed from NSW. *Caveat:* OSM rarely maps 11 kV underground cables leaving zone substations or the "
             "transformers between LV and 11 kV, so a floating LV or 11 kV section usually means *incomplete "
             "mapping*, not a mistake. Floating *transmission* sections are counted as likely errors.\n")
    L.append("**2. Dangling ends.** Original way ends that touch no other way (end-to-end or T-junction), "
             "no substation and no transformer. *Caveat:* a radial LV or 11 kV feeder legitimately ends at its last "
             "pole, so LV / 11 kV / untagged ends are listed as *unconnected ends to check*, not errors. Ends within "
             f"{p['near_miss_m']:g} m of another way are marked `near_miss` (probable unjoined gap, the best "
             "candidates to fix). Transmission lines should end at a substation, so a transmission dangling end is "
             "a likely error (a missing substation or a gap).\n")
    L.append("**3. Missing voltage.** `power=line/minor_line/cable` ways whose `voltage` tag is missing or "
             "unparseable. A tag gap, not a topology error.\n")
    L.append("**4. Voltage conflicts.** A junction (no substation or transformer within the tolerances) where "
             "two ways with *disjoint* known voltage sets meet. Multi-valued tags count as all their voltages, so "
             "`11000;415` meeting `415` is fine (LV underbuilt on 11 kV poles), but `11000` meeting `415` is "
             "flagged. Untagged ways never conflict. Counted per junction. LV meeting 11/22 kV "
             "(`lv_vs_hv_distribution`) is almost always an LV way starting at a pole whose transformer is not "
             "mapped, i.e. missing equipment, so it is a *to check* item, not an error; other pairs (e.g. 22 kV "
             "meeting 11 kV, or transmission meeting distribution) are likely errors.\n")
    L.append("**5. Duplicates and overlaps.** Exact duplicates have identical geometry (either direction); one "
             "row per redundant copy. A substantial overlap is a way lying within "
             f"{p['overlap_buffer_m']:g} m of another for at least {p['overlap_min_m']:g} m and at least "
             f"{p['overlap_min_frac']:.0%} of its own length. Shorter partial overlaps, mostly separately mapped "
             "132 kV circuits converging onto shared towers near substations, are counted but not reported. "
             "Duplicate substations: same name (likely error), or one footprint mostly inside another / a point "
             "inside a footprint (to check: may be a genuine site-within-a-site).\n")

    L.append("## Examples (with OSM links)\n")
    sections = [
        ("Transmission dangling ends", (issues["issue_type"] == "dangling_end") & (issues["line_class"] == "transmission"), None),
        ("Floating components (largest first)", issues["issue_type"] == "floating_component", None),
        ("Voltage conflicts", issues["issue_type"] == "voltage_conflict", None),
        ("Overlapping / duplicate lines", issues["issue_type"].isin(["overlapping_lines", "duplicate_line"]), None),
        ("Duplicate substations", issues["issue_type"] == "duplicate_substation", None),
        ("Distribution near-miss ends (probable gaps)", (issues["issue_type"] == "dangling_end")
         & issues["subtype"].str.endswith("near_miss") & (issues["line_class"] != "transmission"), None),
        ("Longest ways with no voltage tag", issues["issue_type"] == "missing_voltage", None),
    ]
    for title, mask, _ in sections:
        sub = issues[mask]
        if title.startswith("Longest"):
            sub = sub.assign(_l=sub.geometry.length).sort_values("_l", ascending=False)
        if title.startswith("Distribution near"):
            sub = sub.assign(_d=sub["details"].str.extract(r"ends ([\d.]+) m")[0].astype(float)).sort_values("_d")
        L.append(f"### {title} ({len(sub)})\n")
        L.append(_examples(sub) + "\n")
    L.append("## Limitations\n")
    L.append("- OSM power data in the ACT is incomplete (many underground 11 kV cables, transformers and service "
             "lines are missing), so absence in OSM is not absence on the ground.\n"
             "- Results depend on the tolerances above; see the sensitivity table.\n"
             "- Lines that merely cross (no shared OSM node, no end near the other line) are *not* connected, "
             "because overhead lines cross without joining.\n"
             "- Transformers include both `power=transformer` features and pole-top transformers mapped as a "
             "`transformer=*` tag on a `power=pole` node; OSM still maps only a small fraction of the real "
             "distribution transformers, so most LV-vs-11 kV conflicts are unmapped equipment.\n"
             "- osmnx returns whole ways that intersect the ACT, so the whole-way totals include line length in "
             "NSW (mostly 330 kV and 66 kV); use the inside-the-ACT figures for the ACT network.\n"
             f"- Counts by type: {fmt(by)}.\n")
    L.append(f"\n---\n{DATA_NOTE}\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def run(path: Path = PROCESSED_PATH, outputs_dir: Path = OUTPUTS_DIR, topology_path: Path = TOPOLOGY_PATH) -> dict:
    t0 = time.perf_counter()
    lines = gpd.read_file(path, layer=LINES_LAYER)
    subs = gpd.read_file(path, layer=SUBSTATIONS_LAYER)
    fps = gpd.read_file(path, layer=FOOTPRINTS_LAYER)
    txs = gpd.read_file(path, layer=TRANSFORMERS_LAYER)
    boundary = gpd.read_file(path, layer=BOUNDARY_LAYER).to_crs(PROJECT_CRS).geometry.union_all()

    topo = build_graph(lines, subs, fps, txs)
    issues, extra = build_issues(topo, subs, fps, boundary)
    if "source" in txs:
        extra["transformers_by_source"] = {str(k): int(v) for k, v in txs["source"].value_counts().items()}
    fp_only = component_table(build_graph(lines, subs, fps, txs, polygon_point_catchment=False))
    fed_fp_only = round(float(fp_only.loc[fp_only["has_substation"], "length_m"].sum())
                        / float(lines.geometry.length.sum()), 3)

    no_t = build_graph(lines, subs, fps, txs, snap_interior=False)
    nd = find_dangling_ends(no_t, boundary)
    sens = sensitivity_runs(lines, subs, fps, txs, boundary,
                            tols=(2.0, DEFAULT_TOL, 10.0))
    runtime = round(time.perf_counter() - t0, 2)
    summary = summarise(topo, issues, extra, sensitivity=sens,
                        no_tsnap_dangling=int(nd["inside_study_area"].sum()),
                        fed_share_footprint_only=fed_fp_only, runtime_s=runtime)

    # long member lists (big components) are truncated in the files; the
    # ``components`` layer keeps the full membership
    issues = issues.copy()
    issues["related_osm_ids"] = issues["related_osm_ids"].map(_truncate_ids)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    issues.drop(columns="geometry").to_csv(outputs_dir / ISSUES_CSV.name, index=False)
    (outputs_dir / SUMMARY_JSON.name).write_text(json.dumps(summary, indent=2, default=str))
    (outputs_dir / REPORT_MD.name).write_text(render_markdown(summary, issues))

    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.unlink(missing_ok=True)
    issues.to_file(topology_path, layer=ISSUES_LAYER, driver="GPKG")
    comps = extra["components"]
    geoms = topo.lines.geometry.values
    comp_gdf = gpd.GeoDataFrame({
        "component_id": comps["component_id"].astype(int),
        "tier": comps["tier"],
        "top_class": comps["top_class"],
        "n_lines": comps["n_lines"].astype(int),
        "length_km": (comps["length_m"] / 1000).round(3),
        "has_substation": comps["has_substation"],
        "has_transformer": comps["has_transformer"],
        "voltages": [";".join(map(str, v)) for v in comps["voltages"]],
        "substation_osm_ids": [";".join(map(str, s)) for s in comps["substations"]],
        "member_osm_ids": [_ids(topo.lines, idx) for idx in comps["line_idx"]],
    }, geometry=[shapely.multilinestrings(list(geoms[idx])) for idx in comps["line_idx"]],
        crs=topo.lines.crs)
    comp_gdf.to_file(topology_path, layer=COMPONENTS_LAYER, driver="GPKG")
    return summary


def main() -> int:
    summary = run()
    iss = summary["issues"]
    print(f"Issues: {iss['total_rows']} rows ({iss['by_group']}), "
          f"{iss['distinct_osm_objects_flagged']} distinct OSM objects")
    print(f"By type: {iss['by_type']}")
    print(f"Components: {summary['network']['n_components']}, largest = "
          f"{summary['network']['largest_component_share_of_length']:.1%} of length")
    print(f"Wrote {ISSUES_CSV}, {REPORT_MD}, {SUMMARY_JSON}, {TOPOLOGY_PATH} "
          f"in {summary['runtime_s']} s")
    print(DATA_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
