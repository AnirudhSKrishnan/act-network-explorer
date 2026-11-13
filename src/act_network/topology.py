"""Build a connectivity graph of the (OSM) power network.

Graph model
-----------
``build_graph`` returns a :class:`Topology` whose ``graph`` is a
``networkx.MultiGraph`` with three kinds of node:

* **junction** nodes (int ids ``0..k-1``): clusters of line endpoints that lie
  within ``tol`` metres of each other (single-linkage, so a chain of endpoints
  each within ``tol`` of the next becomes one node). Node attrs ``x``/``y`` are
  the mean of the member endpoints (EPSG:7855).
* **substation** nodes ``("sub", osm_id)``.
* **transformer** nodes ``("tx", osm_id)``.

Every OSM way becomes one or more ``kind="line"`` edges. A way is split into
several edges where something attaches to its *interior*:

* **T-junctions** -- in OSM a distribution line very often ends part-way along
  another line (at a pole), not at its endpoint. An endpoint within ``tol`` of
  another line's interior is attached there: the target line is split at the
  projected point and the endpoint's junction node is inserted into it. (If the
  projected point is within ``tol`` of the target's own end, the two endpoint
  clusters are simply merged instead.) A way never snaps onto itself.
  By default a T-snap is NOT made between a transmission line (>= 66 kV) and a
  distribution line (< 66 kV): an 11 kV / LV line ending under a 132 kV line is
  almost always an underbuilt or crossing circuit, not a connection, and
  snapping it would both fake a voltage conflict and hide floating sections.
  Those cases are recorded in ``Topology.blocked_snaps`` instead.
* **shared interior OSM nodes** -- two ways that pass through the same vertex
  (neither ending there) are joined at it, as OSM topology implies, unless
  their voltages are incompatible or it would join transmission to
  distribution (then it is a shared pole carrying separate circuits).
* **devices passed by a line** -- a line whose interior crosses a substation
  footprint (or passes within ``tol`` of a point substation / transformer)
  without either of its endpoints being linked to that device is split there
  and the device node is inserted into the line.

Devices are linked to line endpoints by ``kind="substation_link"`` /
``kind="transformer_link"`` edges (length 0):

* polygon substation: endpoint inside the footprint buffered by
  ``footprint_tol``, or within ``sub_tol`` of its point (the centroid, as in the
  ``substations`` layer) -- the brief's "inside its polygon or within ~30 m of
  its point" rule; the centroid clause only matters for small footprints,
* point substation: endpoint within ``sub_tol`` of the point,
* transformer: endpoint within ``tx_tol`` of the point.

Transformers therefore *connect* (an LV line ending at a transformer is joined
to the 11 kV line feeding it) but are not supply points: a component counts as
"fed" only if it contains a substation node. See ``report.py`` for the checks.

All functions are pure (GeoDataFrames in, objects out); geometry must be in a
projected CRS in metres (EPSG:7855 for this project).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from act_network.clean import TRANSMISSION_MIN_V

DEFAULT_TOL = 5.0  # m, endpoint <-> endpoint / line-interior snapping
DEFAULT_SUB_TOL = 30.0  # m, endpoint <-> point substation
DEFAULT_FOOTPRINT_TOL = 5.0  # m, buffer around substation footprints
DEFAULT_TX_TOL = 30.0  # m, endpoint <-> transformer


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def voltage_set(voltages: str | None) -> frozenset[int]:
    """``"11000;415"`` -> ``frozenset({11000, 415})``; ``""`` / None -> empty set.

    Uses the ``voltages`` column (parsed, ";"-joined ints) rather than
    ``voltage_max`` (which reads back as float with NaN).
    """
    if voltages is None or (not isinstance(voltages, str) and pd.isna(voltages)):
        return frozenset()
    return frozenset(int(float(t)) for t in str(voltages).split(";") if t.strip())


def voltages_compatible(a: frozenset[int], b: frozenset[int]) -> bool:
    """Two lines may meet directly if either voltage is unknown or they share a voltage.

    A multi-valued way such as ``"11000;415"`` (11 kV with LV underbuilt on the
    same poles) is compatible with both an 11 kV and an LV line; ``11000`` meeting
    ``415`` directly (no transformer/substation) is not.
    """
    return not a or not b or bool(a & b)


def _class_of(vset: frozenset[int]) -> str:
    if not vset:
        return "unknown"
    return "transmission" if max(vset) >= TRANSMISSION_MIN_V else "distribution"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:  # path compression
            p[i], i = root, p[i]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    """Result of :func:`build_graph`.

    Attributes
    ----------
    graph: the MultiGraph described in the module docstring.
    lines: the input lines (reset index; row position == ``line_idx``) plus a
        ``vset`` column (frozenset of volts).
    endpoints: one row per original way endpoint: ``line_idx``, ``end``
        ("start"/"end"), ``node`` (junction id), ``x``, ``y``, ``device_links``
        (list of device node ids linked to that junction).
    blocked_snaps: T-snaps not made because they would join transmission and
        distribution directly: ``line_idx`` (source), ``end``, ``target_idx``,
        ``distance_m``.
    tjunctions: T-snaps that were made (same columns as ``blocked_snaps``).
    stats: counts about the snapping (for the report).
    params: tolerances used.
    """

    graph: nx.MultiGraph
    lines: gpd.GeoDataFrame
    endpoints: pd.DataFrame
    blocked_snaps: pd.DataFrame
    tjunctions: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def device_nodes(self, kind: str) -> list:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == kind]

    def line_degree_others(self, node, line_idx: int) -> int:
        """Number of line-edges at ``node`` that belong to ways other than ``line_idx``."""
        return sum(
            1
            for _, _, d in self.graph.edges(node, data=True)
            if d["kind"] == "line" and d["line_idx"] != line_idx
        )

    def node_has_device(self, node) -> bool:
        return any(d["kind"] != "line" for _, _, d in self.graph.edges(node, data=True))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _device_table(
    substations: gpd.GeoDataFrame | None,
    footprints: gpd.GeoDataFrame | None,
    transformers: gpd.GeoDataFrame | None,
    *,
    tol: float,
    sub_tol: float,
    footprint_tol: float,
    tx_tol: float,
    polygon_point_catchment: bool = True,
) -> pd.DataFrame:
    """One row per device: node id, kind, label point, endpoint catchment, interior-attach area."""
    rows = []
    fp_by_id = {}
    if footprints is not None and len(footprints):
        for oid, geom in zip(footprints["osm_id"], footprints.geometry):
            fp_by_id[int(oid)] = geom
    seen_subs: set[int] = set()
    if substations is not None and len(substations):
        for oid, geom in zip(substations["osm_id"], substations.geometry):
            oid = int(oid)
            seen_subs.add(oid)
            poly = fp_by_id.get(oid)
            pt = geom if geom.geom_type == "Point" else geom.centroid
            if poly is not None:
                # footprint (+ footprint_tol) OR within sub_tol of the substation point
                # (centroid); the latter only matters for small footprints
                catchment = poly.buffer(footprint_tol)
                if polygon_point_catchment:
                    catchment = shapely.union(catchment, pt.buffer(sub_tol))
                rows.append((("sub", oid), "substation", pt, catchment, poly))
            else:
                rows.append((("sub", oid), "substation", pt, pt.buffer(sub_tol), pt.buffer(tol)))
    # footprints whose substation row is missing still count
    for oid, poly in fp_by_id.items():
        if oid not in seen_subs:
            rows.append((("sub", oid), "substation", poly.representative_point(),
                         poly.buffer(footprint_tol), poly))
    if transformers is not None and len(transformers):
        for oid, geom in zip(transformers["osm_id"], transformers.geometry):
            pt = geom if geom.geom_type == "Point" else geom.centroid
            rows.append((("tx", int(oid)), "transformer", pt, pt.buffer(tx_tol), pt.buffer(tol)))
    return pd.DataFrame(rows, columns=["node", "kind", "point", "catchment", "attach"])


def build_graph(
    lines: gpd.GeoDataFrame,
    substations: gpd.GeoDataFrame | None = None,
    footprints: gpd.GeoDataFrame | None = None,
    transformers: gpd.GeoDataFrame | None = None,
    *,
    tol: float = DEFAULT_TOL,
    sub_tol: float = DEFAULT_SUB_TOL,
    footprint_tol: float = DEFAULT_FOOTPRINT_TOL,
    tx_tol: float = DEFAULT_TX_TOL,
    snap_interior: bool = True,
    block_cross_class_tsnap: bool = True,
    connect_shared_vertices: bool = True,
    polygon_point_catchment: bool = True,
) -> Topology:
    """Build the network graph (see module docstring for the model).

    ``lines`` needs ``osm_id``, ``voltages`` (";"-joined volts, "" if unknown)
    and LineString geometry; ``line_class`` / ``length_m`` are optional.
    ``snap_interior=False`` turns off T-junction snapping and
    ``polygon_point_catchment=False`` drops the "within ``sub_tol`` of the
    centroid" clause for polygon substations (footprint only); the report uses
    both to show how much they matter.
    """
    lines = lines.reset_index(drop=True).copy()
    if "voltages" not in lines:
        lines["voltages"] = ""
    lines["vset"] = [voltage_set(v) for v in lines["voltages"]]
    geoms = lines.geometry.values
    n_lines = len(lines)
    lengths = shapely.length(geoms)
    vclass = [_class_of(v) for v in lines["vset"]]

    # --- 1. endpoints: index 2*i = start of line i, 2*i+1 = end of line i
    starts = shapely.get_point(geoms, 0)
    ends = shapely.get_point(geoms, -1)
    pts = np.empty(2 * n_lines, dtype=object)
    pts[0::2], pts[1::2] = starts, ends
    pt_line = np.repeat(np.arange(n_lines), 2)
    uf = _UnionFind(2 * n_lines)

    pt_tree = shapely.STRtree(pts)
    a, b = pt_tree.query(pts, predicate="dwithin", distance=tol)
    keep = (a < b) & (pt_line[a] != pt_line[b])  # never merge a way's own two ends
    for i, j in zip(a[keep], b[keep]):
        uf.union(int(i), int(j))
    # closed rings: their two ends are the same point -> same node
    for i in np.flatnonzero(shapely.is_closed(geoms)):
        uf.union(2 * int(i), 2 * int(i) + 1)

    # --- 2. T-junctions: endpoint near another line's interior
    splits: list[tuple[int, float, object]] = []  # (line_idx, dist_along, pt_idx | device node)
    blocked: list[tuple] = []
    made: list[tuple] = []
    if snap_interior and n_lines:
        line_tree = shapely.STRtree(geoms)
        p_idx, l_idx = line_tree.query(pts, predicate="dwithin", distance=tol)
        own = pt_line[p_idx] == l_idx
        p_idx, l_idx = p_idx[~own], l_idx[~own]
        dist_along = shapely.line_locate_point(geoms[l_idx], pts[p_idx])
        for p, li, d in zip(p_idx, l_idx, dist_along):
            p, li = int(p), int(li)
            if d <= tol:
                uf.union(p, 2 * li)  # effectively at the target's start
            elif d >= lengths[li] - tol:
                uf.union(p, 2 * li + 1)  # at the target's end
            else:
                src_cls, tgt_cls = vclass[pt_line[p]], vclass[li]
                if block_cross_class_tsnap and {src_cls, tgt_cls} == {"transmission", "distribution"}:
                    blocked.append((int(pt_line[p]), "start" if p % 2 == 0 else "end", li,
                                    float(shapely.distance(pts[p], geoms[li]))))
                    continue
                splits.append((li, float(d), ("pt", p)))
                made.append((int(pt_line[p]), "start" if p % 2 == 0 else "end", li,
                             float(shapely.distance(pts[p], geoms[li]))))

    roots = np.array([uf.find(i) for i in range(2 * n_lines)])
    uniq, node_of_pt = np.unique(roots, return_inverse=True)
    node_of_pt = node_of_pt.astype(int)
    xs, ys = shapely.get_x(pts), shapely.get_y(pts)

    G = nx.MultiGraph()
    node_x = pd.Series(xs).groupby(node_of_pt).mean()
    node_y = pd.Series(ys).groupby(node_of_pt).mean()
    for n in range(len(uniq)):
        G.add_node(n, kind="junction", x=float(node_x[n]), y=float(node_y[n]))

    # max single-linkage cluster extent (sanity check on chaining)
    cluster_diam = 0.0
    sizes = np.bincount(node_of_pt)
    for n in np.flatnonzero(sizes > 2) if len(sizes) else []:
        m = node_of_pt == n
        cx, cy = xs[m], ys[m]
        d = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2).max()
        cluster_diam = max(cluster_diam, float(d))
    for n in np.flatnonzero(sizes == 2) if len(sizes) else []:
        i, j = np.flatnonzero(node_of_pt == n)
        cluster_diam = max(cluster_diam, float(np.hypot(xs[i] - xs[j], ys[i] - ys[j])))

    # --- 2b. ways that share an interior OSM node (X-junction at a shared pole)
    n_shared, n_shared_skipped = 0, 0
    if connect_shared_vertices and n_lines:
        coords, cidx = shapely.get_coordinates(geoms, return_index=True)
        ncoord = shapely.get_num_coordinates(geoms)
        first = np.r_[0, np.cumsum(ncoord)[:-1]]
        is_end = np.zeros(len(coords), bool)
        is_end[first] = True
        is_end[first + ncoord - 1] = True
        vdf = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "li": cidx, "end": is_end})
        vdf = vdf[~vdf["end"]].drop_duplicates(["x", "y", "li"])
        shared = vdf[vdf.duplicated(["x", "y"], keep=False)]
        next_node = len(uniq)
        for (x, y), grp in shared.groupby(["x", "y"]):
            idx = sorted(set(int(i) for i in grp["li"]))
            classes = {vclass[i] for i in idx}
            ok = all(voltages_compatible(lines["vset"].iat[a], lines["vset"].iat[b])
                     for k, a in enumerate(idx) for b in idx[k + 1:])
            if not ok or (block_cross_class_tsnap and {"transmission", "distribution"} <= classes):
                n_shared_skipped += 1  # shared pole carrying separate circuits
                continue
            G.add_node(next_node, kind="junction", x=float(x), y=float(y))
            pt = shapely.Point(x, y)
            for i in idx:
                splits.append((i, float(shapely.line_locate_point(geoms[i], pt)), next_node))
            next_node += 1
            n_shared += 1

    # --- 3. devices
    dev = _device_table(substations, footprints, transformers, tol=tol, sub_tol=sub_tol,
                        footprint_tol=footprint_tol, tx_tol=tx_tol,
                        polygon_point_catchment=polygon_point_catchment)
    linked_pairs: set[tuple[int, object]] = set()  # (line_idx, device node) linked via an endpoint
    if len(dev):
        for node, kind, pt in zip(dev["node"], dev["kind"], dev["point"]):
            G.add_node(node, kind=kind, x=float(pt.x), y=float(pt.y))
        catch = shapely.STRtree(dev["catchment"].values)
        p_i, d_i = catch.query(pts, predicate="intersects")
        for p, di in zip(p_i, d_i):
            dnode, dkind = dev["node"].iat[di], dev["kind"].iat[di]
            jn = int(node_of_pt[p])
            if not G.has_edge(jn, dnode, key=dkind):
                G.add_edge(jn, dnode, key=dkind, kind=f"{dkind}_link", length_m=0.0)
            linked_pairs.add((int(pt_line[p]), dnode))
        # lines passing through / by a device without an endpoint link to it
        attach = shapely.STRtree(dev["attach"].values)
        l_i, d_i = attach.query(geoms, predicate="intersects")
        for li, di in zip(l_i, d_i):
            li = int(li)
            dnode = dev["node"].iat[di]
            if (li, dnode) in linked_pairs:
                continue
            area = dev["attach"].iat[di]
            where = shapely.intersection(geoms[li], area)
            anchor = where.representative_point() if not where.is_empty else dev["point"].iat[di]
            d = float(shapely.line_locate_point(geoms[li], anchor))
            splits.append((li, d, dnode))
            linked_pairs.add((li, dnode))

    # --- 4. line edges (split at T-junctions / devices)
    by_line: dict[int, list[tuple[float, object]]] = {}
    for li, d, target in splits:
        node = int(node_of_pt[target[1]]) if isinstance(target, tuple) and target[0] == "pt" else target
        by_line.setdefault(li, []).append((d, node))
    osm_ids = lines["osm_id"].to_numpy()
    lclass = lines["line_class"].to_numpy() if "line_class" in lines else np.array(vclass)
    volt_str = lines["voltages"].fillna("").to_numpy()
    for li in range(n_lines):
        seq = [(0.0, int(node_of_pt[2 * li]))]
        seq += sorted(by_line.get(li, []), key=lambda t: t[0])
        seq.append((float(lengths[li]), int(node_of_pt[2 * li + 1])))
        seg = 0
        for (d0, n0), (d1, n1) in zip(seq[:-1], seq[1:]):
            if n0 == n1 and d1 - d0 < 1e-6 and len(seq) > 2:
                continue  # zero-length hop onto the same node
            G.add_edge(n0, n1, key=(li, seg), kind="line", line_idx=li,
                       osm_id=int(osm_ids[li]), voltages=str(volt_str[li]),
                       line_class=str(lclass[li]), length_m=float(d1 - d0))
            seg += 1

    # --- 5. endpoint table
    links_of_node: dict[int, list] = {}
    for u, v, d in G.edges(data=True):
        if d["kind"] != "line":
            j, dn = (u, v) if isinstance(u, (int, np.integer)) else (v, u)
            links_of_node.setdefault(int(j), []).append(dn)
    endpoints = pd.DataFrame({
        "line_idx": pt_line,
        "end": np.tile(["start", "end"], n_lines),
        "node": node_of_pt,
        "x": xs,
        "y": ys,
    })
    endpoints["device_links"] = [links_of_node.get(int(n), []) for n in node_of_pt]

    stats = {
        "n_lines": int(n_lines),
        "n_endpoints": int(2 * n_lines),
        "n_junction_nodes": int(len(uniq)),
        "n_tjunction_snaps": int(len(made)),
        "n_blocked_cross_class_tsnaps": int(len(blocked)),
        "n_device_interior_attachments": int(sum(1 for s in splits if isinstance(s[2], tuple) and s[2][0] in ("sub", "tx"))),
        "n_shared_interior_nodes_joined": int(n_shared),
        "n_shared_interior_nodes_not_joined": int(n_shared_skipped),
        "max_cluster_extent_m": round(cluster_diam, 2),
        "n_substation_nodes": int((dev["kind"] == "substation").sum()) if len(dev) else 0,
        "n_transformer_nodes": int((dev["kind"] == "transformer").sum()) if len(dev) else 0,
    }
    params = {
        "tol_m": tol, "sub_tol_m": sub_tol, "footprint_tol_m": footprint_tol, "tx_tol_m": tx_tol,
        "snap_interior": snap_interior, "block_cross_class_tsnap": block_cross_class_tsnap,
        "connect_shared_vertices": connect_shared_vertices,
        "polygon_point_catchment": polygon_point_catchment,
    }
    snap_cols = ["line_idx", "end", "target_idx", "distance_m"]
    return Topology(G, lines, endpoints, pd.DataFrame(blocked, columns=snap_cols),
                    pd.DataFrame(made, columns=snap_cols), stats, params)


# ---------------------------------------------------------------------------
# Component summary
# ---------------------------------------------------------------------------


def component_table(topo: Topology) -> pd.DataFrame:
    """One row per connected component that contains at least one line.

    Columns: ``component_id`` (0 = largest by line length), ``line_idx`` (list),
    ``n_lines``, ``length_m``, ``substations`` / ``transformers`` (lists of
    osm_ids), ``has_substation``, ``has_transformer``, ``voltages`` (sorted
    desc, all known voltages in the component), ``top_class``
    ("transmission" / "distribution" / "unknown" by the highest known voltage).
    """
    G = topo.graph
    lengths = topo.lines.geometry.length.to_numpy()
    rows = []
    for comp in nx.connected_components(G):
        line_ids: set[int] = set()
        subs, txs = [], []
        for n in comp:
            kind = G.nodes[n].get("kind")
            if kind == "substation":
                subs.append(n[1])
            elif kind == "transformer":
                txs.append(n[1])
        for _, _, d in G.edges(comp, data=True):
            if d["kind"] == "line":
                line_ids.add(d["line_idx"])
        if not line_ids:
            continue
        idx = sorted(line_ids)
        volts = sorted(set().union(*(topo.lines["vset"].iat[i] for i in idx)), reverse=True)
        rows.append({
            "line_idx": idx,
            "n_lines": len(idx),
            "length_m": float(lengths[idx].sum()),
            "substations": sorted(subs),
            "transformers": sorted(txs),
            "has_substation": bool(subs),
            "has_transformer": bool(txs),
            "voltages": volts,
            "top_class": _class_of(frozenset(volts)),
        })
    cols = ["line_idx", "n_lines", "length_m", "substations", "transformers",
            "has_substation", "has_transformer", "voltages", "top_class"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["length_m", "n_lines"], ascending=False).reset_index(drop=True)
    df.insert(0, "component_id", np.arange(len(df)))
    return df
