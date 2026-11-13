# ACT electricity network from OpenStreetMap: summary of numbers

Generated: 2026-11-13

> Data: © OpenStreetMap contributors (ODbL); suburbs © ABS (CC BY 4.0). Public OSM data — NOT Evoenergy network data; OSM power data is incomplete. Service areas are rough geometric estimates, not feeder boundaries.

Every figure below is computed by `python -m act_network.build` from the processed data and check outputs. Nothing here is Evoenergy data, and OSM coverage is incomplete, so these are counts of what is mapped in OpenStreetMap, not of the real network.

## Line length

**Headline: 1,817.7 km of mapped line inside the ACT** (14.3 km of it underground), from 6,804 mapped OSM segments. Lengths are measured in EPSG:7855 and clipped to the ACT boundary (`length_in_act_m`). Overlapping or duplicated ways are counted as mapped.

Why two columns: OpenStreetMap queries return whole ways, and 49 of the segments cross the ACT border. Summing whole-way lengths gives 2,192.2 km (14.4 km underground), which includes NSW sections. Use the in-ACT figures.

By voltage band:

| Voltage band | km inside the ACT | Whole mapped ways incl. NSW sections (km) |
|---|---|---|
| 330 kV | 85.3 | 357.3 |
| 132 kV | 274.0 | 280.8 |
| 66 kV | 24.8 | 91.4 |
| 22 kV | 33.0 | 34.0 |
| 11 kV | 635.7 | 648.9 |
| LV (415 V) | 617.6 | 617.6 |
| unknown | 147.3 | 162.3 |
| **total** | **1,817.7** | 2,192.2 |

By line class (transmission = 66 kV and above; unknown = no usable voltage tag):

| Line class | km inside the ACT | Whole mapped ways incl. NSW sections (km) |
|---|---|---|
| transmission | 384.2 | 729.5 |
| distribution | 1,286.3 | 1,300.5 |
| unknown | 147.3 | 162.3 |

Underground (power=cable or location=underground): **14.3 km** inside the ACT (14.4 km whole mapped ways; 31 segments).

Substation bays and busbars: 114 ways tagged `line=bay` / `line=busbar` (5.24 km inside the ACT, 5.24 km whole ways; 5.24 km transmission inside the ACT). They are conductors inside substation sites, not network lines, but they are included in the totals and bands above. Excluding them, transmission is 378.9 km inside the ACT.

## Substations

Counts of OSM substation **features**, by class. Features are not the same as sites: the same site can be mapped more than once.

| Class | OSM features |
|---|---|
| transmission | 5 |
| zone | 17 |
| distribution | 37 |
| **total** | **59** |

- Zone substations: 17 features, **16 distinct sites** after merging same-name duplicates ("Angle Crossing Zone Substation" is mapped twice). The rough nearest-substation areas (16) use one seed per site. They are a geometric estimate, not feeder boundaries.
- The quality checks also flagged 2 substation features as a duplicate or nested footprint of another feature:
  - same name: substation 'Angle Crossing Zone Substation' has the same name as substation 680650738
  - nested footprint: substation 'Canberra STS' footprint lies mostly inside the footprint of substation 160191425
- 36 substation features have no name tag.

Transformers (OSM point features, not a count of installed transformers): **83** total (43 from power=transformer, 40 from pole transformer tag). `power=transformer` nodes are mapped devices; "pole transformer tag" means the transformer was inferred from a tag on a pole.

## Data-quality issues

The checks flagged **6,044** issue rows involving 5,337 distinct OSM objects (this counts every object involved, including the partner way of an overlap and every member way of a floating section, so it is not a count of wrong objects). The row count is not an error count: most rows are 'to check' items (e.g. dangling line ends and disconnected sections, which may partly reflect incomplete OSM mapping) and tag gaps, not confirmed mistakes. None has been verified against the real network.

By group:

| Group | Rows flagged |
|---|---|
| likely error | 29 |
| to check | 5,389 |
| tag gap | 626 |

Of the 29 rows in the 'likely error' group, **22 are inside substation sites** (overlapping ways within a substation site) and **7 are elsewhere**. These are the ones flagged by the checks as probable mapping problems, not confirmed errors.

Overlapping-lines check: 26 flagged, 22 of them inside 3 substation site(s) and the rest elsewhere.

By check:

| Check | Rows flagged |
|---|---|
| dangling end | 4,358 |
| duplicate substation | 2 |
| floating component | 764 |
| missing voltage | 626 |
| overlapping lines | 26 |
| voltage conflict | 268 |

## Furthest suburbs from the nearest zone substation

Straight-line distance from a suburb's representative point to the nearest zone substation (not along the network).

| # | Suburb | Distance (km) | Nearest zone substation |
|---|---|---|---|
| 1 | Uriarra Village | 13.6 | Latham Zone Substation |
| 2 | Tharwa | 7.4 | Theodore Zone Substation |
| 3 | Whitlam | 6.9 | Civic Zone Substation |
| 4 | Denman Prospect | 5.9 | Woden Substation |
| 5 | Macnamara | 5.6 | Latham Zone Substation |

## Toy 11 kV feeder load flow (assumed loads and cable type; not a real Evoenergy feeder study)

A pandapower toy model on one 11 kV path taken from the OSM data. Route and lengths come from OpenStreetMap; every electrical parameter is assumed (see the assumptions table in [outputs/feeder_results.md](feeder_results.md)). The results below describe the toy model only, not any real feeder.

- Substation: **Fyshwick Zone Substation** (OSM 344074567)
- Path: **8.47 km** along 6 OSM ways, **10 buses** (9 model sections). Trunk only: branches are ignored. Only 2 of 17 zone-substation features touch any line tagged 11 kV, so the choice was narrow.
- Assumed cable: NA2XS2Y 1x185 RM/25 12/20 kV on every section; 9 loads of 200 kW each at power factor 0.95, ideal 1.00 pu source, no transformers.

| Result (toy model) | Base loads | Loads doubled |
|---|---|---|
| Total load (kW) | 1,800 | 3,600 |
| End-of-feeder voltage (pu) | 0.986 | 0.971 |
| End-of-feeder voltage drop (%) | 1.40 | 2.87 |
| Max cable loading (% of rating) | 27.4 | 55.7 |
| Losses (kW) | 16.1 | 66.8 |

Load multiplier (x the base loads) at which the toy model reaches a reference limit: voltage below 0.94 pu at **4.03x**; cable loading above 100 % of rating at **3.52x**. Binding limit: thermal (loading > 100 % of cable rating). The limits are typical-practice references, not Evoenergy's standards.

