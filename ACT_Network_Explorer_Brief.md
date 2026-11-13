# Project brief: ACT Electricity Network Explorer

Prepared 13 November 2026. Copy this file into an empty project folder and start from step 0.

## Why this exists

Anirudh is an RMIT student (Bachelor of Engineering, Electronic and Computer Systems Engineering (Honours), expected June 2028) applying for the **Evoenergy Engineering Summer Vacation Program** in Canberra (EOI, job no. X7EQX). The ad says knowledge of "GIS or ArcFM systems would be advantageous but is not essential" and prefers a power systems focus.

This is a 2–3 day, self-directed project to:
1. learn how GIS is used to manage an electricity distribution network,
2. have something real to show for it on the resume and in the cover letter, and
3. add a small power systems piece (a load flow) to cover the area where his degree is lighter.

## Honesty rules (do not break these)

- It uses **public OpenStreetMap data**. It is not Evoenergy's network data, and OSM power data is incomplete. Say so in the README and on the map.
- It is **not ArcFM experience**. ArcFM is a licensed Schneider Electric product. Describe this as "GIS" or "open-source GIS (QGIS, GeoPandas)".
- Substation "service areas" are a rough geometric estimate, not real feeder boundaries. Label them that way.
- Any load-flow numbers come from a toy model with assumed loads and cable types. Label every assumption.
- No made-up figures on the resume. Use only numbers the project actually produces.

## Stack

- Python 3.11+, `geopandas`, `shapely` 2.x, `osmnx` (2.x), `networkx`, `folium`, `matplotlib`, `pandapower`, `pytest`
- QGIS (free) for looking at the data and making one clean static map
- Coordinate system for any distance or area work: **GDA2020 / MGA zone 55 (EPSG:7855)**. Store raw data in WGS84 (EPSG:4326).

## Data sources

| Data | Source | Notes |
|---|---|---|
| Power lines, cables, substations, poles, towers, transformers | OpenStreetMap via `osmnx.features.features_from_place("Australian Capital Territory, Australia", tags={"power": [...]})` | Tags to pull: `line`, `minor_line`, `cable`, `substation`, `transformer`, `tower`, `pole`. Keep the `voltage`, `name`, `operator` and `substation` tags. Licence: ODbL, credit "© OpenStreetMap contributors" |
| ACT boundary | Same OSM place query (`ox.geocode_to_gdf`) | |
| Suburb boundaries | ABS ASGS 2021 Suburbs and Localities (SAL) shapefile; filter to the ACT (state code 8) | Licence: CC BY 4.0. ACT Government open data (data.act.gov.au / ACTmapi) is an alternative |
| Optional cross-check | Evoenergy's public planning report, if it lists zone substations | Use only to sanity-check names. Don't copy their maps |

Download once and cache raw files in `data/raw/`. Never call the network inside tests.

## Folder layout

```
act-network-explorer/
  README.md
  pyproject.toml           # or requirements.txt
  data/raw/                # cached downloads (gitignored if large)
  data/processed/          # cleaned GeoPackage
  src/act_network/
    fetch.py               # download and cache OSM + ABS data
    clean.py               # parse voltage, split line types, reproject
    topology.py            # build graph, connectivity checks
    service_areas.py       # Voronoi areas, suburb distances
    report.py              # data-quality report (CSV + Markdown)
    webmap.py              # folium interactive map
    feeder.py              # stretch: pandapower toy feeder
  tests/                   # pytest with small synthetic GeoDataFrames
  outputs/                 # map.html, PNGs, quality_report.md
```

## Plan

### Step 0: setup (about 1 hour)
- Make a venv, install packages, `git init`, and add `.gitignore`.
- Create the public GitHub repo `act-network-explorer` under github.com/AnirudhSKrishnan.

### Day 1: get the data and map it
1. `fetch.py`: download the OSM power features, the ACT boundary and the SAL suburbs, and cache them.
2. `clean.py`:
   - Parse `voltage`. It can be `"132000"`, `"66000;11000"` or missing, so keep a list of voltages plus a `voltage_missing` flag.
   - Split the data into transmission lines (66 kV and above), distribution lines (under 66 kV), underground `cable`, and substations.
   - Convert substation points or polygons to one point each (use the centroid if it's a polygon).
   - Save to `data/processed/act_power.gpkg`.
3. Make a first map in QGIS: lines coloured by voltage, substations, suburb outlines.
4. Write down the counts the data gives you, such as total km of line by voltage and number of substations. These are the only numbers to put on the resume later.

### Day 2: network logic and data quality
1. `topology.py`:
   - Build a `networkx` graph from line segments.
   - Snap line endpoints within a tolerance (start with 5 m in EPSG:7855) so near-touching lines connect.
   - Treat a substation as connected when a line endpoint falls inside its polygon or within about 30 m of its point.
2. Checks for `report.py`:
   - connected components with no substation (floating sections of network)
   - dangling line ends not touching another line or a substation
   - lines with a missing voltage tag
   - components that mix voltages without a substation between them
   - duplicate or overlapping geometries
3. `service_areas.py`:
   - Draw Voronoi polygons around zone substations with `shapely.voronoi_polygons`, clipped to the ACT boundary. Label them "rough nearest-substation areas".
   - Measure the distance from each suburb's centroid to the nearest substation, and rank the suburbs furthest away.
4. Write tests with small hand-made GeoDataFrames:
   - a two-line network that should connect after snapping
   - a dangling end
   - a missing voltage
   - a component with no substation
   - Voronoi output count equals substation count
5. Outputs: `quality_report.md` and a CSV of every issue with its OSM id, so someone could go and fix it.

### Day 3: present it, plus the power systems stretch goal
1. `webmap.py`: build a folium map with toggle layers for transmission, distribution, substations, service areas and data-quality issues. Add a popup on each feature showing its OSM id and tags. Put the OSM credit and a "not Evoenergy data" note on the map.
2. README:
   - What it is and why you built it, in one paragraph.
   - Screenshots of the map.
   - How to run it.
   - What the checks find.
   - Limitations: OSM gaps, snapping tolerance, Voronoi isn't feeder boundaries.
3. Stretch goal, `feeder.py`, a pandapower toy radial feeder:
   - Choose one real-looking 11 kV path from the graph and use its segment lengths from the GIS data.
   - Use a standard pandapower cable type, such as `NA2XS2Y 1x185 RM/25 12/20 kV`.
   - Add a few assumed loads, for example 200 kW each at 0.95 power factor.
   - Run `pp.runpp`. Report the end-of-feeder voltage and line loading, plus what happens if the load doubles.
   - Put every assumption in a table in the README.
4. Clean up: run `pytest` until it's green and add a GitHub Actions workflow that runs the tests.

## Done means

- [ ] `python -m act_network.fetch` then the build runs end to end from a clean clone, using cached data
- [ ] `outputs/map.html` opens and every layer toggles
- [ ] `outputs/quality_report.md` lists real issues with OSM ids
- [ ] `pytest` passes offline, and CI is green
- [ ] README states the data source, the licence credits and the limitations
- [ ] (stretch) the feeder load flow runs and its assumptions are listed

## Using it in the application (after it works)

Example resume bullet. Fill in real numbers only:
> Built a GIS tool in Python (GeoPandas, networkx) that maps the ACT's electricity network from public OpenStreetMap data, checks it for connectivity and data-quality problems (found N issues across X km of line), and estimates nearest-substation areas; ran a pandapower load flow on a sample 11 kV feeder.

Cover letter angle: the ad asks what you hope to achieve. Say you built this to learn how a distribution network looks in GIS, what you noticed (for example, how much data quality matters), and that you want to see how Evoenergy does it with real network data and ArcFM.

Also lead with the induction motor digital twin research on the Evoenergy resume. It's the closest thing to power engineering he has.

## Application checklist (Evoenergy)

- Resume (list extracurriculars, as the ad asks)
- Cover letter about you and what you hope to achieve in the program
- Most recent academic transcript (the merged BITS + RMIT transcript)
- Confirmation of work rights (student visa; the ad says student visa holders are welcome)
- Apply through their careers site only. They don't accept emailed applications.
- Program is in Canberra, so be ready to relocate for the summer.
- No closing date found. Last year's listing closed 24 Dec 2025, but don't count on that. Apply as soon as the project is presentable.
