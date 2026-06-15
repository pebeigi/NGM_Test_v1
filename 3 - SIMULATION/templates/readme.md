# templates

Static **SUMO `.net.xml` network skeletons**. The scenario runners in [`models/`](../models/readme.md)
parse a template, reposition nodes/edges to match the lengths chosen in the GUI, inject signal logic,
and write the final network under `models/`. The lane counts in a template's filename must match the
GUI selection so phase strings and lane connections line up.

## Freeway — `freeway_{N}lane_template[_variant].net.xml`

- `{N}` = number of mainline lanes (1–5 in the repo).
- Base `freeway_{N}lane_template.net.xml` → **`on_off`** layout (one on-ramp, one off-ramp, one weaving
  section).
- `_on_on_off_off` → two on-ramps then two off-ramps (longer weaving sequence).
- `_on_off_on_off` → alternating on–off–on–off ramp layout.

## Arterial — `multi_intersections_EW{x}_NS{y}_template.net.xml`

- `EW{x}` = east–west lanes per direction (2 or 3); `NS{y}` = north–south lanes per direction (2 or 3).
- Three signalized intersections along the east–west arterial, with cross-street legs.

## Single intersection

- `single_intersection_EW{x}_NS{y}_template.net.xml` — **signalized** (2–3 lanes each way).
- `single_intersection_EW{x}_NS{y}_stop_sign_template.net.xml` — **all-way stop** (1–2 lanes each way).
- `single_intersection.net.xml` — generic fallback template.

## TGSIM

- `I90_94_simple.net.xml` — real-world I-90/94 style network (6-lane mainline → 3-lane NB/SB diverge),
  used as-is by `tgsim_sim.py`.
