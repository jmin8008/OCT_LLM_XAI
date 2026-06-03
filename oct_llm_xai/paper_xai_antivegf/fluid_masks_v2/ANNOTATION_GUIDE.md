# Fluid Annotation Guide — 12×12 (v2)

You are localizing **retinal fluid** on a pre-injection OCT macular B-scan to a
**12×12 grid**. One eye = one binary 12×12 mask (cells that contain fluid = 1).

## The image
- `grid/<eye>.png`: contrast-stretched OCT crop with a **12×12 green grid**.
  - **Column index 0–11** labelled along the **top**, **row index 0–11** along the
    **left**. A cell is addressed as **[row, col]**, row 0 = top, col 0 = left.
  - There is a 22 px label margin (top + left). Judge fluid by the underlying image,
    not the margin.

## How to read fluid (orientation-agnostic — define relative to the bright band)
Find the **bright hyperreflective band** = retina + RPE complex. Fluid = abnormal
**dark (hyporeflective) spaces** in/around it:
- **SRF (sub-retinal fluid):** dark dome/triangular space **just beneath** the bright
  neurosensory-retina band (band is lifted off, fluid under it).
- **IRF (intra-retinal fluid):** round/oval dark **cysts WITHIN** the bright retinal
  band. Dominant in DME.
- **PED (pigment-epithelial detachment):** dome-shaped **elevation of the RPE** — mark
  the cells under the elevated dome.
- **HRF / drusen alone are NOT fluid.** Do not mark them.

## Constraint (use it!)
Each eye entry lists `fluid_types_present` (ground-truth biomarker labels). **Only
those fluid types exist in this eye.** If it says `["IRF"]`, mark only intraretinal
cysts; ignore anything that looks like SRF. If the list is empty → the mask is all
zero (negative control).

## Output
For each eye, mark the cells overlapping fluid. Be reasonably tight — mark a cell if a
**meaningful part** of it (≳25%) contains fluid. Typical eyes have ~8–25 fluid cells of
144. Do not flood the whole grid.

Write results to your assigned `annot_parts/<batch>.json` as:
```json
{
  "115R": {"fluid_cells": [[2,4],[2,5],[3,4]], "types_seen": ["SRF","PED"], "confidence": "med", "note": "subfoveal SRF dome + nasal PED"},
  "...":  {...}
}
```
- `fluid_cells`: list of [row,col], 0-indexed, row/col in 0..11.
- `confidence`: "high" | "med" | "low" (low if the crop is ambiguous/low-quality).
- Keep `note` short (≤12 words).

Annotate every eye in your batch. If an image fails to load, set
`{"fluid_cells": [], "confidence": "low", "note": "image unreadable"}`.
