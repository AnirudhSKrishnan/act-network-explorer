"""Static matplotlib map -> ``outputs/map.png`` (stand-in for the QGIS map).

Run with::

    python -m act_network.static_map

Data: (c) OpenStreetMap contributors (ODbL); suburbs (c) ABS (CC BY 4.0). Public
OSM data, NOT official network operator data; OSM power data is incomplete.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from act_network import clean, service_areas  # noqa: E402
from act_network.webmap import BAND_COLOURS, BAND_WEIGHTS, DISCLAIMER, SUB_STYLE  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MAP_PNG = OUTPUTS_DIR / "map.png"
DPI = 150
TITLE = "ACT electricity network from OpenStreetMap: lines by voltage band and substations"
# Draw low voltages first so higher voltages sit on top.
BAND_ORDER = ["LV (415 V)", "unknown", "other", "11 kV", "22 kV", "66 kV", "132 kV", "330 kV"]
LEGEND_BANDS = ["330 kV", "132 kV", "66 kV", "22 kV", "11 kV", "LV (415 V)", "unknown"]
# Line widths are a bit thinner than on the web map (print scale).
_W_SCALE = 0.6


def plot_map(lines, substations, suburbs, boundary, areas=None, dpi: int = DPI):
    """Return a matplotlib Figure. Inputs are in a projected CRS (EPSG:7855)."""
    fig, ax = plt.subplots(figsize=(8, 11.5))
    suburbs.boundary.plot(ax=ax, color="#BBBBBB", linewidth=0.4, zorder=1)
    if areas is not None and len(areas):
        areas.boundary.plot(ax=ax, color="#777777", linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

    bands = lines["voltage_band"].fillna("unknown")
    for band in BAND_ORDER:
        sub = lines[bands == band]
        if len(sub):
            sub.plot(ax=ax, color=BAND_COLOURS[band], linewidth=BAND_WEIGHTS[band] * _W_SCALE, zorder=4)

    for cls in ("distribution", "zone", "transmission"):
        s = substations[substations["sub_class"] == cls]
        if len(s):
            colour, radius = SUB_STYLE[cls]
            s.plot(ax=ax, color=colour, edgecolor="black", linewidth=0.5, markersize=radius * 5, zorder=6)
    zone = substations[substations["sub_class"] == "zone"].drop_duplicates("name")
    for _, r in zone.iterrows():
        if isinstance(r["name"], str):
            label = r["name"].replace(" Zone Substation", "").replace(" Substation", "").replace(" Switch Station", "")
            ax.annotate(label, (r.geometry.x, r.geometry.y), xytext=(4, 3), textcoords="offset points",
                        fontsize=5.5, color="#222222", zorder=7)

    handles = [Line2D([], [], color=BAND_COLOURS[b], linewidth=max(1.2, BAND_WEIGHTS[b]), label=b)
               for b in LEGEND_BANDS]
    handles += [
        Line2D([], [], marker="o", linestyle="", markeredgecolor="black", markerfacecolor=SUB_STYLE[c][0],
               markersize=SUB_STYLE[c][1] * 0.9, label=f"{c.capitalize()} substation")
        for c in ("transmission", "zone", "distribution")
    ]
    handles += [Line2D([], [], color="#BBBBBB", linewidth=0.8, label="Suburb boundary"),
                Line2D([], [], color="black", linewidth=1.2, label="ACT boundary")]
    if areas is not None and len(areas):
        handles.append(Line2D([], [], color="#777777", linewidth=0.8, linestyle=(0, (4, 3)),
                              label="Rough nearest-substation area\n(not a feeder boundary)"))
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.92, title="Legend", title_fontsize=8)

    # Crop the view to the ACT (plus a margin); OSM lines that run on into NSW are cut off by the view.
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 3_000
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_title(TITLE, fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (m, GDA2020 / MGA zone 55, EPSG:7855)", fontsize=7)
    ax.set_ylabel("Northing (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(style="plain")
    fig.text(0.5, 0.012, "\n".join(textwrap.wrap(DISCLAIMER, 120)), ha="center", va="bottom",
             fontsize=6.5, color="#333333")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> int:
    p = clean.PROCESSED_PATH
    lines = gpd.read_file(p, layer=clean.LINES_LAYER)
    subs = gpd.read_file(p, layer=clean.SUBSTATIONS_LAYER)
    suburbs = gpd.read_file(p, layer=clean.SUBURBS_LAYER)
    boundary = gpd.read_file(p, layer=clean.BOUNDARY_LAYER)
    areas = gpd.read_file(service_areas.SERVICE_AREAS_PATH, layer=service_areas.AREAS_LAYER)
    fig = plot_map(lines, subs, suburbs, boundary, areas)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAP_PNG, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {MAP_PNG} ({MAP_PNG.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
