"""Clean the raw OSM / ABS data into ``data/processed/act_power.gpkg``.

Run with::

    python -m act_network.clean

All output layers are in EPSG:7855 (GDA2020 / MGA zone 55) so lengths come out
in metres. Lists cannot be stored in a GeoPackage column, so multi-valued
fields (e.g. voltages) are ";"-joined strings. See the layer constants below;
downstream modules import them rather than hard-coding layer names.

The cleaning steps are pure functions of a (raw) GeoDataFrame so they can be
unit-tested without touching ``data/raw/``; ``main()`` does the file I/O.

Data provenance: public OpenStreetMap data (c) OpenStreetMap contributors
(ODbL) -- NOT Evoenergy's network data, and OSM power mapping is incomplete.
Suburb boundaries: ABS ASGS 2021 (CC BY 4.0).
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from act_network import fetch

PROJECT_CRS = 7855  # GDA2020 / MGA zone 55

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "act_power.gpkg"

LINES_LAYER = "lines"
SUBSTATIONS_LAYER = "substations"
FOOTPRINTS_LAYER = "substation_footprints"
TRANSFORMERS_LAYER = "transformers"
SUPPORTS_LAYER = "supports"
BOUNDARY_LAYER = "act_boundary"
SUBURBS_LAYER = "suburbs"

LINE_POWER_TAGS = ("line", "minor_line", "cable")

# Voltage (V) at/above which a line or substation counts as transmission-level
# for the line_class rule, and the boundary used to decide whether a substation
# "steps down to distribution voltage".
TRANSMISSION_MIN_V = 66_000
# Voltage at/above which a substation is classed as transmission outright.
SUBSTATION_TRANSMISSION_MIN_V = 330_000

# Exact-voltage band labels (by voltage_max).
_BAND_LABELS = {
    330_000: "330 kV",
    132_000: "132 kV",
    66_000: "66 kV",
    22_000: "22 kV",
    11_000: "11 kV",
}
LV_BAND = "LV (415 V)"
LV_MAX_V = 1_000  # anything up to 1 kV (415 V, 240 V, ...) is low voltage

# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

# One voltage token: a number with an optional "kV" / "V" unit, e.g. "132000",
# "11kV", "11 kV", "0.415 kV", "415 V".
_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(kv|v)?$", re.IGNORECASE)


def parse_voltage(value: str | None) -> list[int]:
    """Parse an OSM ``voltage`` tag into a list of volts (ints).

    Behaviour (the tag is free text, so be forgiving but never guess):

    * ``None`` / NaN / ``""`` / whitespace            -> ``[]``
    * ``"132000"``                                    -> ``[132000]``
    * ``"66000;11000"``                               -> ``[66000, 11000]``
      (result is de-duplicated and sorted descending, whatever the tag order)
    * a ``kV`` unit is honoured: ``"11kV"`` -> ``[11000]``; a bare ``V`` unit
      is just stripped: ``"415 V"`` -> ``[415]``
    * junk tokens (``"unknown"``, ``"high"``, ``"0"``, ``"-5"``) are dropped
      individually, so ``"11000;unknown"`` -> ``[11000]``
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    volts: set[int] = set()
    for token in str(value).split(";"):
        m = _TOKEN_RE.match(token.strip())
        if not m:
            continue
        number = float(m.group(1))
        if (m.group(2) or "").lower() == "kv":
            number *= 1000
        if number > 0:
            volts.add(round(number))
    return sorted(volts, reverse=True)


def voltage_band(voltage_max) -> str:
    """Label for a voltage_max (V): "330 kV" ... "11 kV", "LV (415 V)", "other", "unknown"."""
    if voltage_max is None or pd.isna(voltage_max):
        return "unknown"
    v = int(voltage_max)
    if v in _BAND_LABELS:
        return _BAND_LABELS[v]
    if v <= LV_MAX_V:
        return LV_BAND
    return "other"  # e.g. 33 kV / 6.6 kV: real but not one of the ACT bands


def line_class(voltage_max) -> str:
    """"transmission" (>= 66 kV), "distribution" (< 66 kV) or "unknown" (no voltage)."""
    if voltage_max is None or pd.isna(voltage_max):
        return "unknown"
    return "transmission" if int(voltage_max) >= TRANSMISSION_MIN_V else "distribution"


def normalise_operator(value: str | None) -> str | None:
    """Collapse the inconsistent OSM operator spellings.

    Evoenergy / evoenergy / ActewAGL / ACTEWAGL -> "Evoenergy";
    TransGrid / Transgrid -> "TransGrid"; anything else is kept as written
    (whitespace-stripped); empty -> None.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "")
    if key in {"evoenergy", "actewagl"}:
        return "Evoenergy"
    if key == "transgrid":
        return "TransGrid"
    return text


def strip_act_suffix(name: str) -> str:
    """"Acton (ACT)" -> "Acton" (ABS appends " (ACT)" to names shared with other states)."""
    return re.sub(r"\s*\(ACT\)\s*$", "", str(name)).strip()


_STS_RE = re.compile(r"\bSTS\b|Terminal", re.IGNORECASE)
_ZONE_NAME_RE = re.compile(r"Zone Substation|Switch Station", re.IGNORECASE)


def substation_class(name: str | None, volts: list[int]) -> str:
    """Classify a substation as "transmission", "zone" or "distribution".

    The OSM ``substation`` tag is unreliable (zone substations are tagged
    distribution / substation / switchyard), so use voltages and the name:

    * transmission: max voltage >= 330 kV, or name has "STS" / "Terminal"
    * zone: has a voltage >= 66 kV AND a voltage < 66 kV (steps down to
      distribution level), or name has "Zone Substation" / "Switch Station"
    * distribution: everything else (incl. untagged / minor_distribution)
    """
    name = "" if name is None or pd.isna(name) else str(name)
    vmax = max(volts) if volts else None
    if (vmax is not None and vmax >= SUBSTATION_TRANSMISSION_MIN_V) or _STS_RE.search(name):
        return "transmission"
    steps_down = any(v >= TRANSMISSION_MIN_V for v in volts) and any(
        v < TRANSMISSION_MIN_V for v in volts
    )
    if steps_down or _ZONE_NAME_RE.search(name):
        return "zone"
    return "distribution"


def _to_project_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(PROJECT_CRS)


def _text(series: pd.Series) -> pd.Series:
    """Object column with NaN -> None (clean nulls in the GPKG)."""
    return series.astype(object).where(series.notna(), None)


def _add_voltage_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add voltages / voltage_max / voltage_missing derived from the raw voltage tag."""
    parsed = gdf["voltage"].map(parse_voltage)
    gdf["voltages"] = parsed.map(lambda vs: ";".join(map(str, vs)))
    gdf["voltage_max"] = pd.array(
        [max(vs) if vs else pd.NA for vs in parsed], dtype="Int64"
    )
    gdf["voltage_missing"] = parsed.map(len).eq(0).to_numpy()
    return gdf


# ---------------------------------------------------------------------------
# Layer builders (raw GeoDataFrame in -> processed GeoDataFrame out)
# ---------------------------------------------------------------------------

LINE_COLUMNS = [
    "osm_id", "element", "power", "voltage", "voltages", "voltage_max",
    "voltage_missing", "line_class", "voltage_band", "is_underground",
    "name", "operator", "circuits", "cables", "location", "length_m", "geometry",
]


def clean_lines(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """power = line / minor_line / cable, exploded to single LineStrings, EPSG:7855."""
    gdf = raw[raw["power"].isin(LINE_POWER_TAGS)].copy()
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    gdf = _to_project_crs(gdf)
    # Explode any MultiLineString so each row is one LineString (osm_id repeats).
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geom_type == "LineString"].reset_index(drop=True)

    gdf = _add_voltage_columns(gdf)
    gdf["line_class"] = gdf["voltage_max"].map(line_class)
    gdf["voltage_band"] = gdf["voltage_max"].map(voltage_band)
    location = gdf["location"].fillna("").str.strip().str.lower()
    gdf["is_underground"] = ((gdf["power"] == "cable") | (location == "underground")).to_numpy()
    gdf["operator"] = gdf["operator"].map(normalise_operator)
    gdf["length_m"] = gdf.geometry.length

    for col in ("voltage", "name", "circuits", "cables", "location", "element", "power"):
        gdf[col] = _text(gdf[col])
    return gdf[LINE_COLUMNS]


SUBSTATION_COLUMNS = [
    "osm_id", "element", "name", "operator", "substation", "voltage", "voltages",
    "voltage_max", "voltage_missing", "osm_geom_type", "sub_class", "is_zone", "geometry",
]


def _classified_substations(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """power = substation rows with all attribute columns, original geometry, EPSG:7855."""
    gdf = raw[raw["power"] == "substation"].copy()
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    gdf = _to_project_crs(gdf).reset_index(drop=True)
    gdf["osm_geom_type"] = gdf.geom_type
    gdf = _add_voltage_columns(gdf)
    gdf["operator"] = gdf["operator"].map(normalise_operator)
    gdf["sub_class"] = [
        substation_class(n, parse_voltage(v)) for n, v in zip(gdf["name"], gdf["voltage"])
    ]
    gdf["is_zone"] = (gdf["sub_class"] == "zone").to_numpy()
    for col in ("name", "substation", "voltage", "element"):
        gdf[col] = _text(gdf[col])
    return gdf[SUBSTATION_COLUMNS]


def clean_substations(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One Point per power=substation (polygons -> centroid). Duplicates are kept."""
    gdf = _classified_substations(raw)
    gdf["geometry"] = gdf.geometry.centroid  # no-op for points
    return gdf


def clean_substation_footprints(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The polygon substations, same columns as ``substations`` but polygon geometry."""
    gdf = _classified_substations(raw)
    return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)


def clean_transformers(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """power = transformer as points (osm_id, element, name, operator)."""
    gdf = raw[raw["power"] == "transformer"].copy()
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    gdf = _to_project_crs(gdf).reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.centroid  # no-op for points
    gdf["operator"] = gdf["operator"].map(normalise_operator)
    for col in ("name", "element"):
        gdf[col] = _text(gdf[col])
    return gdf[["osm_id", "element", "name", "operator", "geometry"]]


def clean_supports(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Poles / towers (osm_id, power), EPSG:7855."""
    gdf = raw[~(raw.geometry.isna() | raw.geometry.is_empty)]
    return _to_project_crs(gdf)[["osm_id", "power", "geometry"]].reset_index(drop=True)


def clean_boundary(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The ACT boundary as a single polygon (geometry only), EPSG:7855."""
    gdf = _to_project_crs(raw)
    return gpd.GeoDataFrame(geometry=[gdf.geometry.union_all()], crs=PROJECT_CRS)


def clean_suburbs(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """SAL_CODE21, suburb_name (" (ACT)" stripped), is_remainder, geometry; EPSG:7855."""
    gdf = _to_project_crs(raw)
    names = gdf["SAL_NAME21"].map(strip_act_suffix)
    out = gpd.GeoDataFrame(
        {
            "SAL_CODE21": gdf["SAL_CODE21"].astype(str),
            "suburb_name": names,
            "is_remainder": names.str.startswith("ACT Remainder").to_numpy(),
        },
        geometry=gdf.geometry.values,
        crs=PROJECT_CRS,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def main() -> int:
    network = gpd.read_file(fetch.NETWORK_PATH, layer=fetch.NETWORK_LAYER)
    supports = gpd.read_file(fetch.SUPPORTS_PATH, layer=fetch.SUPPORTS_LAYER)
    boundary = gpd.read_file(fetch.BOUNDARY_PATH, layer=fetch.BOUNDARY_LAYER)
    suburbs = gpd.read_file(fetch.SUBURBS_PATH, layer=fetch.SUBURBS_LAYER)

    layers = {
        LINES_LAYER: clean_lines(network),
        SUBSTATIONS_LAYER: clean_substations(network),
        FOOTPRINTS_LAYER: clean_substation_footprints(network),
        TRANSFORMERS_LAYER: clean_transformers(network),
        SUPPORTS_LAYER: clean_supports(supports),
        BOUNDARY_LAYER: clean_boundary(boundary),
        SUBURBS_LAYER: clean_suburbs(suburbs),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.unlink(missing_ok=True)  # rebuild from scratch, no stale layers
    for name, gdf in layers.items():
        gdf.to_file(PROCESSED_PATH, layer=name, driver="GPKG")
        print(f"[{name}] {len(gdf)} rows, CRS EPSG:{gdf.crs.to_epsg()}")
    print(f"Wrote {PROCESSED_PATH}")
    print("Data: (c) OpenStreetMap contributors (ODbL); ABS ASGS (CC BY 4.0). "
          "Public OSM data - not Evoenergy network data; OSM is incomplete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
