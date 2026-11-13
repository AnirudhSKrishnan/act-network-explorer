"""Stretch goal: a pandapower TOY radial 11 kV feeder built from real OSM segment lengths.

**This is a toy model.** The route and the section lengths come from public
OpenStreetMap data (lengths measured in GDA2020 / MGA zone 55, EPSG:7855). The
cable type, the loads, the power factor and the source voltage are all
*assumptions* (see :data:`ASSUMPTIONS`). It is not an Evoenergy feeder study
and says nothing about how the real feeder is loaded or built.

Pipeline (``python -m act_network.feeder``)
-------------------------------------------
1. Build the network graph with :func:`act_network.topology.build_graph` (same
   call as ``report.py``).
2. :func:`select_feeder_path` picks one 11 kV path (rule below).
3. :func:`merge_short_sections` merges very short consecutive sections so the
   model has a readable number of buses.
4. :func:`build_feeder_net` builds a radial pandapower net; :func:`run_load_flow`
   runs it at a load multiplier (1x base, 2x doubled); :func:`find_limit_multiplier`
   bisects for the multiplier where a voltage or thermal limit is first crossed.
5. Writes ``outputs/feeder_results.json``, ``outputs/feeder_results.md`` and
   ``outputs/feeder_profile.png``.

Path selection rule (deterministic)
-----------------------------------
* Work on the **11 kV subgraph**: ``kind="line"`` edges whose parsed voltage set
  contains 11000 V (so ``"11000"`` and ``"11000;415"``), with parallel edges
  collapsed to the shortest one. Lines with an unknown voltage are *not* used.
* For each zone substation node (``is_zone``), the **sources** are the
  substation node itself (if a line edge touches it) plus the junctions linked
  to it by ``substation_link`` edges, restricted to the 11 kV subgraph.
* Other substations are barriers: their nodes and the junctions linked to them
  (and not to the source substation) are removed, so a "feeder" never runs
  through a second substation.
* Multi-source Dijkstra by edge length (m). The candidate end is the node with
  the **largest shortest-path distance** (ties: smaller ``repr`` of the node id).
* Across substations pick the candidate with the largest distance (ties:
  smaller substation OSM id).
* Fallback (only if no zone substation touches an 11 kV edge): in each 11 kV
  component take a double-sweep longest shortest path (exact on trees, a lower
  bound otherwise) and keep the longest; ``fallback=True`` is recorded.

The result is the *trunk* only: branches leaving the path (and whatever load
they carry) are ignored, which understates the real trunk current.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp

from act_network.topology import voltage_set

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
RESULTS_JSON = OUTPUT_DIR / "feeder_results.json"
RESULTS_MD = OUTPUT_DIR / "feeder_results.md"
PROFILE_PNG = OUTPUT_DIR / "feeder_profile.png"

FEEDER_VOLTAGE_V = 11_000
VN_KV = 11.0
CABLE_STD_TYPE = "NA2XS2Y 1x185 RM/25 12/20 kV"
LOAD_KW = 200.0
POWER_FACTOR = 0.95  # lagging
SOURCE_VM_PU = 1.0
MIN_SECTION_M = 200.0  # sections shorter than this are merged into a neighbour
V_MIN_PU = 0.94  # typical-practice -6 % planning reference (NOT Evoenergy's standard)
V_MAX_PU = 1.06
LOADING_LIMIT_PCT = 100.0
DOUBLED = 2.0

TOY_HEADER = "Toy model — assumed loads and cable type; not a real Evoenergy feeder study"

SELECTION_RULE = (
    "From each zone substation, follow only line edges whose voltage tag includes 11000 V "
    "(other substations are barriers); take the node with the longest shortest-path distance "
    "(by GIS length) from the substation; across substations keep the longest "
    "(ties: smaller node id / substation OSM id)."
)


# ---------------------------------------------------------------------------
# Path selection (pure, works on a topology.build_graph MultiGraph)
# ---------------------------------------------------------------------------


def _is_device(node) -> bool:
    return isinstance(node, tuple)


def eleven_kv_subgraph(graph: nx.MultiGraph, voltage_v: int = FEEDER_VOLTAGE_V) -> nx.Graph:
    """Simple graph of the ``kind="line"`` edges whose voltage set contains ``voltage_v``.

    Parallel edges between the same two nodes are collapsed to the shortest one
    (edge attributes copied from the MultiGraph edge, including ``key``).
    """
    H = nx.Graph()
    for u, v, k, d in graph.edges(keys=True, data=True):
        if d.get("kind") != "line" or voltage_v not in voltage_set(d.get("voltages")):
            continue
        if u == v:
            continue
        if H.has_edge(u, v) and H[u][v]["length_m"] <= d["length_m"]:
            continue
        H.add_edge(u, v, key=k, **d)
    return H


def _linked(graph: nx.MultiGraph, node, kind: str = "substation_link") -> set:
    """Nodes joined to ``node`` by a device-link edge of ``kind``."""
    out = set()
    for _, other, d in graph.edges(node, data=True):
        if d.get("kind") == kind:
            out.add(other)
    return out


def _substation_reach(graph: nx.MultiGraph, H: nx.Graph, sub) -> dict | None:
    """Longest shortest 11 kV path from substation ``sub`` (None if it touches no 11 kV edge)."""
    if sub not in graph:
        return None
    own_links = _linked(graph, sub)
    blocked = set()
    for n, d in graph.nodes(data=True):
        if d.get("kind") == "substation" and n != sub:
            blocked.add(n)
            blocked |= {j for j in _linked(graph, n) if j not in own_links}
    Hs = H.subgraph([n for n in H if n not in blocked])
    sources = sorted(({sub} | own_links) & set(Hs.nodes), key=repr)
    if not sources:
        return None
    dist, paths = nx.multi_source_dijkstra(Hs, sources, weight="length_m")
    far = min(dist, key=lambda n: (-dist[n], repr(n)))
    return {"substation_node": sub, "distance_m": float(dist[far]), "path": paths[far],
            "n_sources": len(sources), "subgraph": Hs}


def _double_sweep(H: nx.Graph, comp: set) -> tuple[float, list]:
    start = min(comp, key=repr)
    d1 = nx.single_source_dijkstra_path_length(H, start, weight="length_m")
    a = min(d1, key=lambda n: (-d1[n], repr(n)))
    d2, p2 = nx.single_source_dijkstra(H, a, weight="length_m")
    b = min(d2, key=lambda n: (-d2[n], repr(n)))
    return float(d2[b]), p2[b]


def _sections(H: nx.Graph, path: list) -> list[dict]:
    out = []
    for a, b in zip(path[:-1], path[1:]):
        d = H[a][b]
        out.append({"from_node": a, "to_node": b, "osm_id": int(d["osm_id"]),
                    "line_idx": int(d["line_idx"]), "voltages": d.get("voltages", ""),
                    "length_m": float(d["length_m"])})
    return out


def select_feeder_path(graph: nx.MultiGraph, zone_substations,
                       voltage_v: int = FEEDER_VOLTAGE_V) -> dict:
    """Pick one 11 kV path (see module docstring for the rule).

    ``zone_substations``: iterable of substation node ids (``("sub", osm_id)``).
    Returns a dict with ``substation_node`` (None on fallback), ``path`` (node
    list, starting at the substation side), ``sections`` (one dict per edge:
    ``osm_id``, ``line_idx``, ``voltages``, ``length_m``, ``from_node``,
    ``to_node``), ``total_length_m``, ``fallback``, ``n_zone_substations``,
    ``n_zone_substations_with_11kv``, ``branch_nodes`` (path nodes with more than
    two 11 kV edges, i.e. branches the toy model ignores) and ``rule``.
    """
    H = eleven_kv_subgraph(graph, voltage_v)
    zones = sorted(set(zone_substations), key=repr)
    reaches = [r for r in (_substation_reach(graph, H, s) for s in zones) if r is not None]
    result = {"rule": SELECTION_RULE, "n_zone_substations": len(zones),
              "n_zone_substations_with_11kv": len(reaches),
              "zone_substations_with_11kv": [r["substation_node"] for r in reaches]}
    best = None
    if reaches:
        best = min(reaches, key=lambda r: (-r["distance_m"], repr(r["substation_node"])))
    if best is not None and best["distance_m"] > 0:
        Hs, path = best["subgraph"], best["path"]
        result.update(substation_node=best["substation_node"], fallback=False)
    else:
        if H.number_of_edges() == 0:
            raise ValueError("graph has no line edges at the requested voltage")
        cands = [(*_double_sweep(H, c), c) for c in nx.connected_components(H)]
        dist, path, _ = min(cands, key=lambda t: (-t[0], repr(min(t[2], key=repr))))
        Hs = H
        result.update(substation_node=None, fallback=True)
    sections = _sections(Hs, path)
    comp = nx.node_connected_component(Hs, path[0])
    result.update(
        reachable_11kv_length_m=float(sum(d["length_m"] for _, _, d in Hs.subgraph(comp).edges(data=True))),
        path=list(path),
        sections=sections,
        total_length_m=float(sum(s["length_m"] for s in sections)),
        branch_nodes=[n for n in path[1:-1] if H.degree(n) > 2],
    )
    return result


def merge_short_sections(lengths_m, min_len_m: float = MIN_SECTION_M) -> list[list[int]]:
    """Group consecutive section indices so (almost) every group is >= ``min_len_m``.

    A section shorter than ``min_len_m`` joins the group before it; a short first
    group joins the next one. Returns a list of index lists (order preserved,
    every index used once).
    """
    groups: list[list[int]] = []
    totals: list[float] = []
    for i, length in enumerate(lengths_m):
        if groups and length < min_len_m:
            groups[-1].append(i)
            totals[-1] += length
        else:
            groups.append([i])
            totals.append(float(length))
    if len(groups) > 1 and totals[0] < min_len_m:
        groups[1] = groups[0] + groups[1]
        groups.pop(0)
    return groups


# ---------------------------------------------------------------------------
# pandapower model (pure)
# ---------------------------------------------------------------------------


def q_kvar_for(p_kw: float, pf: float) -> float:
    """Reactive power (kvar, positive = lagging / inductive) for ``p_kw`` at power factor ``pf``."""
    return p_kw * math.tan(math.acos(pf))


def build_feeder_net(section_lengths_km, load_kw: float = LOAD_KW, pf: float = POWER_FACTOR,
                     load_buses=None, std_type: str = CABLE_STD_TYPE, vn_kv: float = VN_KV,
                     vm_pu: float = SOURCE_VM_PU, name: str = "toy 11 kV feeder"):
    """Radial feeder: bus 0 (ext_grid, ``vm_pu``) -- line 0 -- bus 1 -- ... -- bus n.

    ``load_buses``: bus indices that get one ``load_kw`` / ``pf`` (lagging) load
    each; default every bus except the source bus 0. No transformer is modelled.
    """
    lengths = [float(x) for x in section_lengths_km]
    if not lengths or min(lengths) <= 0:
        raise ValueError("need at least one section, all lengths > 0")
    net = pp.create_empty_network(name=name)
    if not pp.std_type_exists(net, std_type, "line"):
        raise ValueError(f"pandapower has no line std type {std_type!r}")
    buses = [pp.create_bus(net, vn_kv=vn_kv, name=f"bus {i}") for i in range(len(lengths) + 1)]
    pp.create_ext_grid(net, buses[0], vm_pu=vm_pu, va_degree=0.0, name="zone substation 11 kV bus")
    for i, length in enumerate(lengths):
        pp.create_line(net, buses[i], buses[i + 1], length_km=length, std_type=std_type,
                       name=f"section {i + 1}")
    if load_buses is None:
        load_buses = buses[1:]
    q_kvar = q_kvar_for(load_kw, pf)
    for b in load_buses:
        pp.create_load(net, b, p_mw=load_kw / 1000.0, q_mvar=q_kvar / 1000.0, name=f"load @ bus {b}")
    return net


def run_load_flow(net, load_multiplier: float = 1.0) -> dict:
    """Run ``pp.runpp`` with every load scaled by ``load_multiplier``; return plain-float results.

    Distances along the feeder are cumulative line lengths from bus 0 (the
    model is a chain built by :func:`build_feeder_net`). On non-convergence
    returns ``{"converged": False, ...}``. The loads' ``scaling`` column is
    restored afterwards (the ``res_*`` tables keep this run's results).
    """
    old_scaling = net.load["scaling"].copy()
    net.load["scaling"] = float(load_multiplier)
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="flat", numba=False)
        return _collect_results(net, load_multiplier)
    except pp.LoadflowNotConverged:
        return {"converged": False, "load_multiplier": float(load_multiplier)}
    finally:
        net.load["scaling"] = old_scaling


def _collect_results(net, load_multiplier: float) -> dict:
    vm = net.res_bus.vm_pu.sort_index().to_numpy()
    dist_km = np.r_[0.0, np.cumsum(net.line.sort_index().length_km.to_numpy())]
    end = int(len(vm) - 1)
    loading = net.res_line.loading_percent
    total_load_kw = float((net.load.p_mw * net.load.scaling).sum() * 1000)
    total_load_kvar = float((net.load.q_mvar * net.load.scaling).sum() * 1000)
    return {
        "converged": True,
        "load_multiplier": float(load_multiplier),
        "total_load_kw": total_load_kw,
        "total_load_kvar": total_load_kvar,
        "end_bus": end,
        "end_vm_pu": float(vm[end]),
        "end_voltage_drop_pct": float((1.0 - vm[end] / vm[0]) * 100),
        "min_vm_pu": float(vm.min()),
        "min_vm_bus": int(vm.argmin()),
        "max_vm_pu": float(vm.max()),
        "max_line_loading_pct": float(loading.max()),
        "max_loading_line": int(loading.idxmax()),
        "max_line_current_a": float(net.res_line.i_ka.max() * 1000),
        "total_losses_kw": float(net.res_line.pl_mw.sum() * 1000),
        "source_p_kw": float(net.res_ext_grid.p_mw.sum() * 1000),
        "source_q_kvar": float(net.res_ext_grid.q_mvar.sum() * 1000),
        "bus_vm_pu": [float(x) for x in vm],
        "bus_distance_km": [float(x) for x in dist_km],
        "line_loading_pct": [float(x) for x in loading.sort_index()],
    }


def voltage_violated(res: dict, v_min: float = V_MIN_PU, v_max: float = V_MAX_PU) -> bool:
    return (not res["converged"]) or res["min_vm_pu"] < v_min or res["max_vm_pu"] > v_max


def thermal_violated(res: dict, limit_pct: float = LOADING_LIMIT_PCT) -> bool:
    return (not res["converged"]) or res["max_line_loading_pct"] > limit_pct


def find_limit_multiplier(net, violated, lo: float = 1.0, hi: float = 20.0,
                          tol: float = 0.01) -> float | None:
    """Smallest load multiplier in ``[lo, hi]`` where ``violated(result)`` is True (bisection).

    Non-convergence counts as violated. Returns ``lo`` if already violated at
    ``lo`` and None if not violated at ``hi``. Assumes monotonic behaviour in
    the multiplier (true for voltage drop and current on a radial feeder with
    constant-power loads up to voltage collapse). Leaves loads at scaling 1.
    """
    try:
        if violated(run_load_flow(net, lo)):
            return float(lo)
        if not violated(run_load_flow(net, hi)):
            return None
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if violated(run_load_flow(net, mid)):
                hi = mid
            else:
                lo = mid
        return float(hi)
    finally:
        net.load["scaling"] = 1.0


# ---------------------------------------------------------------------------
# Assumptions (single source for the JSON and the Markdown table)
# ---------------------------------------------------------------------------


def assumptions(n_loads: int, std: dict) -> list[dict]:
    q = q_kvar_for(LOAD_KW, POWER_FACTOR)
    return [
        {"item": "Model status", "value": "Toy model",
         "basis": "Assumed loads and cable type; not an Evoenergy feeder study and not Evoenergy data."},
        {"item": "Route and section lengths", "value": "OpenStreetMap 11 kV ways",
         "basis": "Lengths measured on the OSM geometry in GDA2020 / MGA zone 55 (EPSG:7855). "
                  "OSM is incomplete and may not match the real feeder route or switching."},
        {"item": "Path selection rule", "value": "Longest shortest 11 kV path from a zone substation",
         "basis": SELECTION_RULE},
        {"item": "Section merging", "value": f"sections < {MIN_SECTION_M:.0f} m merged",
         "basis": "Very short OSM sections joined to the previous one (first one to the next) "
                  "so the model has a readable number of buses. Total length unchanged."},
        {"item": "Trunk only", "value": "branches ignored",
         "basis": "Branches leaving the path and their load are not modelled, so real trunk "
                  "current would be higher than shown."},
        {"item": "Source", "value": f"ext_grid, {SOURCE_VM_PU:.2f} pu at the zone substation 11 kV bus",
         "basis": "Assumed stiff source at nominal voltage. Real zone substations regulate the "
                  "11 kV bus with tap changers, often above 1.00 pu."},
        {"item": "Transformer", "value": "not modelled",
         "basis": "No 132(66)/11 kV zone transformer and no 11 kV/415 V distribution transformers; "
                  "loads sit directly on 11 kV buses."},
        {"item": "Nominal voltage", "value": f"{VN_KV:g} kV", "basis": "From the OSM voltage tag."},
        {"item": "Conductor", "value": f"{CABLE_STD_TYPE} (all sections)",
         "basis": f"pandapower standard type (r = {std['r_ohm_per_km']} Ω/km, x = {std['x_ohm_per_km']} Ω/km, "
                  f"c = {std['c_nf_per_km']} nF/km, rating {std['max_i_ka'] * 1000:.0f} A). Chosen "
                  "because the project brief suggests it and it ships with pandapower: a common European "
                  "185 mm² Al XLPE 12/20 kV cable. Most of this OSM path is overhead (power=line/minor_line, "
                  "one short power=cable section) and real ACT 11 kV conductors differ; overhead lines "
                  "have higher reactance, lower capacitance and different ratings."},
        {"item": "Load size", "value": f"{LOAD_KW:.0f} kW per load",
         "basis": "Assumed round number, not measured demand."},
        {"item": "Power factor", "value": f"{POWER_FACTOR} lagging ({q:.1f} kvar per load)",
         "basis": "Assumed."},
        {"item": "Load placement", "value": f"one load at every bus except the source ({n_loads} loads)",
         "basis": "Assumed: one load at the far end of each (merged) section."},
        {"item": "Load model", "value": "constant power (PQ), balanced 3-phase",
         "basis": "Positive-sequence load flow (pp.runpp, Newton-Raphson); no unbalance, no voltage dependence."},
        {"item": "Doubled-load case", "value": f"all loads x{DOUBLED:g}", "basis": "Uniform scaling of every load."},
        {"item": "Voltage planning reference", "value": f"±6 % ({V_MIN_PU}–{V_MAX_PU} pu)",
         "basis": "Typical-practice reference only, not Evoenergy's standard. In practice the 11 kV "
                  "network gets only part of the end-to-end voltage budget (the LV network and "
                  "distribution transformer also drop voltage)."},
        {"item": "Thermal reference", "value": f"{LOADING_LIMIT_PCT:.0f} % of the std-type max_i_ka",
         "basis": "pandapower's rating for that cable under its reference installation conditions; "
                  "no derating."},
    ]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def plot_profile(base: dict, doubled: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    c_base, c_dbl = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for res, color, label in ((base, c_base, "Base load (1x)"), (doubled, c_dbl, "Doubled load (2x)")):
        x, y = res["bus_distance_km"], res["bus_vm_pu"]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label, zorder=3,
                markeredgecolor=surface, markeredgewidth=1)
        ax.annotate(f"{y[-1]:.3f} pu", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=ink)
    ax.axhline(V_MIN_PU, color=muted, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(0.02, V_MIN_PU + 0.0012, f"{V_MIN_PU} pu: −6 % typical-practice reference (not Evoenergy's)",
            transform=ax.get_yaxis_transform(), fontsize=8, color=muted, va="bottom")
    lo = min(V_MIN_PU, min(doubled["bus_vm_pu"])) - 0.01
    ax.set_ylim(lo, 1.005)
    xmax = base["bus_distance_km"][-1]
    ax.set_xlim(-0.1, xmax * 1.12)
    ax.set_xlabel("Distance from zone substation along the feeder (km)", color=ink)
    ax.set_ylabel("Bus voltage (pu of 11 kV)", color=ink)
    ax.set_title(title, fontsize=10.5, color=ink, loc="left")
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8.5)
    leg = ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.text(0.01, 0.01, f"Toy model: assumed loads ({LOAD_KW:.0f} kW, {POWER_FACTOR} PF lagging) and cable type; "
             "route/lengths from OpenStreetMap. Not Evoenergy data.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def _fmt_mult(x):
    return "not reached by 20x" if x is None else f"{x:.2f}x"


def write_markdown(res: dict, path: Path) -> None:
    p, b, d = res["path"], res["results"]["base"], res["results"]["doubled"]
    lim = res["limits"]
    sub = p["substation_name"] or "(no zone substation: fallback)"
    L = [f"# Toy 11 kV feeder load flow (pandapower)", "",
         f"> **{TOY_HEADER}.**", "> Route and section lengths come from public OpenStreetMap data "
         "(© OpenStreetMap contributors, ODbL); everything electrical is assumed.", "",
         "## Chosen path", "",
         f"- **Source:** {sub} (OSM {p['substation_osm_id']}, tagged voltages {p['substation_voltages']})"
         if not p["fallback"] else "- **Fallback:** no zone substation touches an 11 kV edge; longest 11 kV path used.",
         f"- **Rule:** {SELECTION_RULE}",
         f"- Only {p['n_zone_substations_with_11kv']} of {p['n_zone_substations']} zone-substation features in the "
         "graph touch any edge tagged 11 kV, so the choice was narrow. This counts only voltage-tagged edges: "
         f"{p['n_lines_without_voltage']} lines with no voltage tag are excluded and some of them are probably 11 kV (an OSM data-coverage finding).",
         f"- **Length:** {p['total_length_m'] / 1000:.2f} km over {p['n_osm_sections']} OSM sections on "
         f"{len(p['osm_way_ids'])} OSM ways, merged into {p['n_model_sections']} model sections "
         f"({p['n_model_sections'] + 1} buses).",
         f"- **OSM way ids (substation → end):** {', '.join(str(i) for i in p['osm_way_ids'])}",
         f"- {len(p['branch_nodes'])} nodes along the path have 11 kV branches that the model ignores (trunk only): "
         f"the modelled trunk is {p['total_length_m'] / 1000:.2f} km of the {p['reachable_11kv_length_m'] / 1000:.2f} km "
         "of 11 kV-tagged line reachable from the substation in the graph.",
         "", "| Model section | From bus → to bus | Length (m) | OSM way ids | OSM power tag |",
         "|---|---|---:|---|---|"]
    for s in p["model_sections"]:
        L.append(f"| {s['section']} | {s['section'] - 1} → {s['section']} | {s['length_m']:.0f} | "
                 f"{', '.join(map(str, s['osm_ids']))} | {', '.join(s['power'])} |")
    L += ["", "## Assumptions", "", "| # | Assumption | Value | Basis / caveat |", "|---:|---|---|---|"]
    for i, a in enumerate(res["assumptions"], 1):
        L.append(f"| {i} | {a['item']} | {a['value']} | {a['basis']} |")
    L += ["", "## Results", "",
          "| Metric | Base (1x) | Doubled (2x) |", "|---|---:|---:|",
          f"| Total load | {b['total_load_kw']:.0f} kW / {b['total_load_kvar']:.0f} kvar | "
          f"{d['total_load_kw']:.0f} kW / {d['total_load_kvar']:.0f} kvar |",
          f"| End-of-feeder voltage | {b['end_vm_pu']:.4f} pu | {d['end_vm_pu']:.4f} pu |",
          f"| End-of-feeder voltage drop | {b['end_voltage_drop_pct']:.2f} % | {d['end_voltage_drop_pct']:.2f} % |",
          f"| Minimum bus voltage | {b['min_vm_pu']:.4f} pu (bus {b['min_vm_bus']}) | "
          f"{d['min_vm_pu']:.4f} pu (bus {d['min_vm_bus']}) |",
          f"| Max line loading | {b['max_line_loading_pct']:.1f} % (section {b['max_loading_line'] + 1}, "
          f"{b['max_line_current_a']:.0f} A) | {d['max_line_loading_pct']:.1f} % (section "
          f"{d['max_loading_line'] + 1}, {d['max_line_current_a']:.0f} A) |",
          f"| Total line losses | {b['total_losses_kw']:.1f} kW | {d['total_losses_kw']:.1f} kW |",
          f"| Within ±6 % reference? | {'yes' if b['within_voltage_reference'] else 'no'} | "
          f"{'yes' if d['within_voltage_reference'] else 'no'} |",
          f"| Within thermal reference? | {'yes' if b['within_thermal_reference'] else 'no'} | "
          f"{'yes' if d['within_thermal_reference'] else 'no'} |",
          "", "### How far could load grow (uniform scaling of all loads)?", "",
          f"- Minimum voltage first falls below {V_MIN_PU} pu at about **{_fmt_mult(lim['voltage_multiplier'])}** "
          "the base load.",
          f"- The most loaded section first exceeds {LOADING_LIMIT_PCT:.0f} % of the cable rating at about "
          f"**{_fmt_mult(lim['thermal_multiplier'])}** the base load.",
          f"- Binding limit in this toy model: **{lim['binding']}**.",
          "", "![Voltage profile](feeder_profile.png)", "",
          "## Reading these numbers", "",
          "- The voltages and loadings are what this assumed model gives, not measurements of the real feeder.",
          "- Loads, power factor, cable type and source voltage are assumptions; change any of them and the "
          "numbers change. Only the route and section lengths come from data (OSM, which is incomplete).",
          "- Branch load is ignored, so real trunk current and voltage drop would be higher for the same "
          "per-bus load.",
          "", "Regenerate with `python -m act_network.feeder`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _node_label(n):
    return f"{n[0]}:{n[1]}" if isinstance(n, tuple) else int(n)


def main() -> int:
    import geopandas as gpd

    from act_network.clean import (FOOTPRINTS_LAYER, LINES_LAYER, PROCESSED_PATH, PROJECT_CRS,
                                   SUBSTATIONS_LAYER, TRANSFORMERS_LAYER)
    from act_network.topology import build_graph

    lines = gpd.read_file(PROCESSED_PATH, layer=LINES_LAYER)
    subs = gpd.read_file(PROCESSED_PATH, layer=SUBSTATIONS_LAYER)
    fps = gpd.read_file(PROCESSED_PATH, layer=FOOTPRINTS_LAYER)
    txs = gpd.read_file(PROCESSED_PATH, layer=TRANSFORMERS_LAYER)
    for g in (lines, subs, fps, txs):
        assert g.crs is not None and g.crs.to_epsg() == PROJECT_CRS, "expected EPSG:7855 layers"
    topo = build_graph(lines, subs, fps, txs)
    G = topo.graph

    zone = subs[subs["is_zone"].astype(bool)]
    zone_nodes = [("sub", int(o)) for o in zone["osm_id"]]
    sel = select_feeder_path(G, zone_nodes)
    sections = sel["sections"]
    power_of = topo.lines["power"].to_numpy() if "power" in topo.lines else None
    for s in sections:
        s["power"] = str(power_of[s["line_idx"]]) if power_of is not None else "?"

    sub_node = sel["substation_node"]
    sub_row = subs[subs["osm_id"].astype(int) == sub_node[1]].iloc[0] if sub_node else None

    groups = merge_short_sections([s["length_m"] for s in sections])
    model_sections = []
    for k, grp in enumerate(groups, 1):
        segs = [sections[i] for i in grp]
        model_sections.append({
            "section": k,
            "length_m": float(sum(s["length_m"] for s in segs)),
            "osm_ids": list(dict.fromkeys(s["osm_id"] for s in segs)),
            "power": list(dict.fromkeys(s["power"] for s in segs)),
            "osm_section_indices": grp,
        })
    lengths_km = [m["length_m"] / 1000 for m in model_sections]

    net = build_feeder_net(lengths_km)
    std = pp.load_std_type(net, CABLE_STD_TYPE, "line")
    base = run_load_flow(net, 1.0)
    doubled = run_load_flow(net, DOUBLED)
    for r in (base, doubled):
        if not r["converged"]:
            raise RuntimeError(f"load flow did not converge at x{r['load_multiplier']}")
        r["within_voltage_reference"] = not voltage_violated(r)
        r["within_thermal_reference"] = not thermal_violated(r)
    v_mult = find_limit_multiplier(net, voltage_violated)
    t_mult = find_limit_multiplier(net, thermal_violated)
    if v_mult is None and t_mult is None:
        binding = "neither limit reached by 20x"
    elif t_mult is None or (v_mult is not None and v_mult <= t_mult):
        binding = f"voltage (min voltage < {V_MIN_PU} pu)"
    else:
        binding = f"thermal (loading > {LOADING_LIMIT_PCT:.0f} % of cable rating)"

    # bus coordinates (EPSG:7855) at the far end of each model section
    path_nodes = sel["path"]
    bus_nodes = [path_nodes[0]] + [path_nodes[g[-1] + 1] for g in groups]
    bus_xy = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in bus_nodes]

    n_loads = len(net.load)
    path_info = {
        "substation_name": None if sub_row is None else str(sub_row["name"]),
        "substation_osm_id": None if sub_node is None else int(sub_node[1]),
        "substation_voltages": None if sub_row is None else str(sub_row["voltages"]),
        "fallback": sel["fallback"],
        "selection_rule": sel["rule"],
        "n_zone_substations": sel["n_zone_substations"],
        "n_zone_substations_with_11kv": sel["n_zone_substations_with_11kv"],
        "zone_substations_with_11kv": [_node_label(n) for n in sel["zone_substations_with_11kv"]],
        "total_length_m": sel["total_length_m"],
        "reachable_11kv_length_m": sel["reachable_11kv_length_m"],
        "n_lines_without_voltage": int((lines["voltages"].fillna("").astype(str) == "").sum()),
        "n_osm_sections": len(sections),
        "n_model_sections": len(model_sections),
        "n_buses": len(model_sections) + 1,
        "osm_way_ids": list(dict.fromkeys(s["osm_id"] for s in sections)),
        "osm_sections": [{**{k: v for k, v in s.items() if k not in ("from_node", "to_node")},
                          "from_node": _node_label(s["from_node"]), "to_node": _node_label(s["to_node"])}
                         for s in sections],
        "model_sections": model_sections,
        "branch_nodes": [_node_label(n) for n in sel["branch_nodes"]],
        "bus_xy_epsg7855": bus_xy,
        "crs": f"EPSG:{PROJECT_CRS}",
    }
    out = {
        "title": TOY_HEADER,
        "data_source": "OpenStreetMap (© OpenStreetMap contributors, ODbL); not Evoenergy data",
        "path": path_info,
        "model": {"std_type": CABLE_STD_TYPE, "std_type_params": std, "vn_kv": VN_KV,
                  "source_vm_pu": SOURCE_VM_PU, "load_kw": LOAD_KW, "power_factor": POWER_FACTOR,
                  "load_kvar": q_kvar_for(LOAD_KW, POWER_FACTOR), "n_loads": n_loads,
                  "load_buses": [int(b) for b in net.load.bus], "pandapower_version": pp.__version__},
        "assumptions": assumptions(n_loads, std),
        "reference_limits": {"v_min_pu": V_MIN_PU, "v_max_pu": V_MAX_PU,
                             "loading_limit_pct": LOADING_LIMIT_PCT,
                             "note": "typical-practice references, not Evoenergy's standards"},
        "results": {"base": base, "doubled": doubled},
        "limits": {"voltage_multiplier": v_mult, "thermal_multiplier": t_mult, "binding": binding,
                   "search_range": [1.0, 20.0], "tolerance": 0.01},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(_jsonable(out), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out, RESULTS_MD)
    name = path_info["substation_name"] or "longest 11 kV path"
    plot_profile(base, doubled, PROFILE_PNG,
                 f"Toy 11 kV feeder from {name}: voltage profile "
                 f"({path_info['total_length_m'] / 1000:.1f} km, {n_loads} x {LOAD_KW:.0f} kW)")

    print(f"Path: {name}, {path_info['total_length_m'] / 1000:.2f} km, {len(sections)} OSM sections -> "
          f"{len(model_sections)} model sections, OSM ways {path_info['osm_way_ids']}")
    for label, r in (("base", base), ("doubled", doubled)):
        print(f"  {label:8s} end {r['end_vm_pu']:.4f} pu ({r['end_voltage_drop_pct']:.2f} % drop), "
              f"min {r['min_vm_pu']:.4f} pu, max loading {r['max_line_loading_pct']:.1f} %, "
              f"losses {r['total_losses_kw']:.1f} kW")
    print(f"  limits: voltage {_fmt_mult(v_mult)}, thermal {_fmt_mult(t_mult)} -> binding: {binding}")
    print(f"Wrote {RESULTS_JSON}, {RESULTS_MD}, {PROFILE_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
