"""Download and cache OSM and ABS data into ``data/raw/``.

Run with::

    python -m act_network.fetch [--force]

Each download is skipped if its cache file already exists, unless ``--force``.

Data provenance and honesty notes
---------------------------------
* The power-network data is **public OpenStreetMap data**. It is NOT
  Evoenergy's network data, and OSM power mapping is incomplete.
  Credit: (c) OpenStreetMap contributors, licensed under the ODbL.
* Suburb boundaries are ABS ASGS Edition 3 (2021) Suburbs and Localities,
  GDA2020, licensed CC BY 4.0 (c) Australian Bureau of Statistics.

Output layout (all in ``data/raw/``)
------------------------------------
* ``act_boundary.gpkg``        layer ``act_boundary`` (EPSG:4326)
* ``osm_power_network.gpkg``   single layer ``power_network`` with mixed
  geometry types (points/lines/polygons), EPSG:4326
* ``osm_power_supports.gpkg``  single layer ``power_supports`` (tower/pole
  nodes), EPSG:4326
* ``act_suburbs.gpkg``         layer ``act_suburbs`` in the ABS native CRS
  (EPSG:7844, GDA2020)
* ``SAL_2021_AUST_GDA2020_SHP.zip``  raw ABS download (cached)
* ``osmnx_cache/``             osmnx HTTP response cache

OSM features carry columns ``element`` (node/way/relation) and ``osm_id``
(reset from the osmnx MultiIndex), plus the tag columns in ``KEEP_TAGS``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests

PLACE = "Australian Capital Territory, Australia"

# data/raw relative to the repo root (src/act_network/fetch.py -> repo root)
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OSMNX_CACHE_DIR = RAW_DIR / "osmnx_cache"

BOUNDARY_PATH = RAW_DIR / "act_boundary.gpkg"
NETWORK_PATH = RAW_DIR / "osm_power_network.gpkg"
SUPPORTS_PATH = RAW_DIR / "osm_power_supports.gpkg"
SUBURBS_PATH = RAW_DIR / "act_suburbs.gpkg"
SAL_ZIP_PATH = RAW_DIR / "SAL_2021_AUST_GDA2020_SHP.zip"

BOUNDARY_LAYER = "act_boundary"
NETWORK_LAYER = "power_network"
SUPPORTS_LAYER = "power_supports"
SUBURBS_LAYER = "act_suburbs"

NETWORK_TAGS = ["line", "minor_line", "cable", "substation", "transformer"]
SUPPORT_TAGS = ["tower", "pole"]

KEEP_TAGS = [
    "power", "voltage", "name", "operator", "substation", "cables",
    "circuits", "frequency", "location", "line", "ref",
]

# Found on the ABS ASGS Edition 3 "Digital boundary files" page
# (https://www.abs.gov.au/statistics/standards/
#  australian-statistical-geography-standard-asgs/
#  edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files)
SAL_URL = (
    "https://www.abs.gov.au/statistics/standards/"
    "australian-statistical-geography-standard-asgs/"
    "edition-3-july-2021-june-2026/access-and-downloads/"
    "digital-boundary-files/SAL_2021_AUST_GDA2020_SHP.zip"
)
USER_AGENT = "act-network-explorer/0.1 (student GIS project)"


def _setup_osmnx():
    """Configure osmnx lazily (no network at import time)."""
    import osmnx as ox

    OSMNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OSMNX_CACHE_DIR)
    ox.settings.requests_timeout = 600
    ox.settings.log_console = False
    return ox


def _should_skip(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"[skip]       {path.name} already exists (use --force to refetch)")
        return True
    return False


def _stringify(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, default=str) if isinstance(value, dict) else ";".join(map(str, value))
    return value


def tidy_osm_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make an osmnx features GeoDataFrame GPKG-safe.

    Resets the (element, id) MultiIndex into ``element`` and ``osm_id`` columns,
    keeps only geometry + ``KEEP_TAGS`` (stringifying any list/dict values),
    drops empty geometries, and keeps EPSG:4326.
    """
    gdf = gdf.reset_index()
    rename = {}
    if "element_type" in gdf.columns:
        rename["element_type"] = "element"
    if "id" in gdf.columns:
        rename["id"] = "osm_id"
    gdf = gdf.rename(columns=rename)
    cols = ["element", "osm_id"] + [c for c in KEEP_TAGS if c in gdf.columns] + ["geometry"]
    gdf = gdf[cols].copy()
    for c in cols:
        if c != "geometry" and gdf[c].dtype == object:
            gdf[c] = gdf[c].map(_stringify)
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    gdf = gdf.set_crs(4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326)
    return gdf.reset_index(drop=True)


def fetch_boundary(force: bool = False) -> Path:
    if _should_skip(BOUNDARY_PATH, force):
        return BOUNDARY_PATH
    ox = _setup_osmnx()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gdf = ox.geocode_to_gdf(PLACE).to_crs(4326)
    gdf = gdf.drop(columns=[c for c in gdf.columns if gdf[c].dtype == object and c != "geometry"
                            and gdf[c].map(lambda v: isinstance(v, (list, dict))).any()])
    gdf.to_file(BOUNDARY_PATH, layer=BOUNDARY_LAYER, driver="GPKG")
    print(f"[downloaded] {BOUNDARY_PATH.name}: {len(gdf)} feature(s)")
    return BOUNDARY_PATH


def _with_retries(fn, attempts: int = 5, wait: float = 30.0):
    """Call ``fn``; on failure (e.g. Overpass 502/429/timeouts) retry with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = wait * 2**i
            print(f"[retry]      {type(exc).__name__}: {str(exc).splitlines()[0][:120]} "
                  f"- waiting {delay:.0f}s (attempt {i + 2}/{attempts})", file=sys.stderr)
            time.sleep(delay)


def _fetch_power_group(tags: list[str], path: Path, layer: str, force: bool) -> Path:
    if _should_skip(path, force):
        return path
    ox = _setup_osmnx()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[querying]   OSM power={tags} (may take several minutes)...")
    raw = _with_retries(lambda: ox.features.features_from_place(PLACE, tags={"power": tags}))
    gdf = tidy_osm_features(raw)
    gdf.to_file(path, layer=layer, driver="GPKG")
    print(f"[downloaded] {path.name}: {len(gdf)} features "
          f"({gdf.geometry.geom_type.value_counts().to_dict()})")
    return path


def fetch_power(force: bool = False) -> list[Path]:
    """Fetch OSM power features as two separate queries (network, supports).

    Failures in one group are reported but do not stop the other.
    """
    out, failures = [], []
    for tags, path, layer in (
        (NETWORK_TAGS, NETWORK_PATH, NETWORK_LAYER),
        (SUPPORT_TAGS, SUPPORTS_PATH, SUPPORTS_LAYER),
    ):
        try:
            out.append(_fetch_power_group(tags, path, layer, force))
        except Exception as exc:  # keep going; one Overpass timeout must not lose the rest
            failures.append(path.name)
            print(f"[FAILED]     {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if failures:
        raise RuntimeError(f"power fetch failed for: {failures}")
    return out


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0 " + USER_AGENT}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
        print(f"[downloaded] {dest.name}: {done / 1e6:.1f} MB" + (f" (expected {total / 1e6:.1f} MB)" if total else ""))
    tmp.replace(dest)


def fetch_suburbs(force: bool = False) -> Path:
    """ABS ASGS 2021 SAL (GDA2020), filtered to the ACT, non-empty geometry only.

    Saved in the ABS native CRS (EPSG:7844). CC BY 4.0, (c) ABS.
    """
    if _should_skip(SUBURBS_PATH, force):
        return SUBURBS_PATH
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not SAL_ZIP_PATH.exists():
        _download(SAL_URL, SAL_ZIP_PATH)
    else:
        print(f"[skip]       {SAL_ZIP_PATH.name} already downloaded")
    gdf = gpd.read_file(f"zip://{SAL_ZIP_PATH}")
    print(f"             SAL columns: {list(gdf.columns)}; {len(gdf)} rows Australia-wide; CRS {gdf.crs}")
    gdf = gdf[gdf["STE_CODE21"].astype(str) == "8"]
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)].reset_index(drop=True)
    gdf.to_file(SUBURBS_PATH, layer=SUBURBS_LAYER, driver="GPKG")
    print(f"[saved]      {SUBURBS_PATH.name}: {len(gdf)} ACT suburbs/localities, CRS {gdf.crs}")
    return SUBURBS_PATH


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch and cache ACT OSM + ABS data into data/raw/")
    p.add_argument("--force", action="store_true", help="re-download even if cache files exist")
    args = p.parse_args(argv)
    print("Data: (c) OpenStreetMap contributors (ODbL); ABS ASGS (CC BY 4.0). "
          "Public OSM data - not Evoenergy network data; OSM is incomplete.")
    steps = (("boundary", fetch_boundary), ("power", fetch_power), ("suburbs", fetch_suburbs))
    failed = []
    for name, fn in steps:
        try:
            fn(force=args.force)
        except Exception as exc:
            failed.append(name)
            print(f"[FAILED]     {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
