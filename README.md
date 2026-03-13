# Rock stacking – 2D packing pipeline

Load rock profiles from a CSV, pack them into a 2D wall using [StablePacking2D](StablePacking2D-master/), and export placements (rock ID, x, y, angle) plus diagnostic plots. No Rhino dependency; can be run standalone or used alongside the Grasshopper/Rhino workflow.

---

## Setup

```bash
cd /path/to/RockStacking
pipenv install
```

Uses Python 3.10 and: `numpy`, `matplotlib`, `scipy`, `opencv-python`, `scikit-image`, `trimesh`, `mosek`.  
If MOSEK license is missing, the pipeline still runs (stability check is bypassed so placement continues).

---

## Quick start

```bash
pipenv run python main_pack.py
```

With explicit wall size and scale (e.g. match Rhino):

```bash
pipenv run python main_pack.py --wall-height 200 --wall-width 175 --pixel-scale 5
```

Output:

- **`rock_placements.csv`** – one row per rock: `rock_id`, `x`, `y` (world coords), `angle`, `placed`
- **`plots/`** – input checks and packing results (see [Plots](#plots))

---

## Input: `data_rocks.csv`

CSV with two columns:

| Column    | Meaning |
|-----------|--------|
| `rock_id` | Integer ID for the rock |
| `profile` | String that evaluates to a list of `[x, y]` points (closed or open polygon) |

Example:

```csv
rock_id,profile
0,"[[-17.56, 55.12], [-17.57, 55.12], ...]"
1,"[[-35.31, 54.91], ...]"
```

Profiles are rasterized at `pixel_scale` pixels per world unit (same convention as Rhino). If a stone would exceed the wall size, it is scaled down to fit.

---

## Output: `rock_placements.csv`

| Column   | Meaning |
|----------|--------|
| `rock_id`| From input |
| `x`      | World x (same scale as input; `x_pixel / pixel_scale`) |
| `y`      | World y, y-up (`(wall_height_px - y_pixel) / pixel_scale`) |
| `angle`  | Placement angle in degrees |
| `placed` | 1 if placed, 0 if unplaced |

Coordinates match the Rhino `result_to_curves` / transformation convention so you can use this CSV to place geometry back in Rhino (or elsewhere).

---

## Command-line options

Run `pipenv run python main_pack.py --help` for the full list. Main ones:

| Option | Default | Description |
|--------|---------|-------------|
| `--wall-height` | 200 | Wall height in **pixels** (same as Rhino). |
| `--wall-width`  | 175 | Wall width in **pixels**. |
| `--pixel-scale` | 5.0 | Pixels per world unit; used for rasterization and for output x, y. |
| `--weight-height` | 0.5 | Height weight in the packer. **Higher (0.5–1.0)** = prefer ground → stones **spread** along the wall. **Low (e.g. 0.03)** = prefer staying close to existing stones → more **clustering**. |
| `--sequence` | `largest_first` | `largest_first` = place biggest stones first (better spread). `random` = shuffle order (uses `--seed`). |
| `--seed` | 0 | Random seed when `--sequence random`. |
| `--target-max-dim` | (auto) | Max pixel dimension per stone; auto = `min(wall_height, wall_width) // 4`. |
| `--data-csv` | `data_rocks.csv` | Input CSV path. |
| `--output-csv` | `rock_placements.csv` | Output placements path. |
| `--plots-dir` | `plots` | Directory for plot images. |
| `--no-plots` | false | Skip writing plots. |

Examples:

```bash
# Spread stones along the wall (defaults)
pipenv run python main_pack.py --wall-height 200 --wall-width 175 --pixel-scale 5

# Rhino-style (more clustering, random order)
pipenv run python main_pack.py --wall-height 200 --wall-width 175 --pixel-scale 5 --weight-height 0.03 --sequence random

# Strong preference for ground (max spread)
pipenv run python main_pack.py --weight-height 1.0
```

---

## Plots

Generated under `plots/` (unless `--no-plots`):

| File | Content |
|------|--------|
| `00_raw_profiles.png` | **Raw CSV outlines** – one subplot per rock, [x,y] polygons before any rasterization. Use to check that input data is correct. |
| `01_input_profiles.png` | **Rasterized stones** used by the packer (after scaling and axis-align). |
| `02_wall.png` | Binary packed wall (occupied vs empty). |
| `03_wall_by_stone.png` | Packed wall with each stone in a different color. |
| `04_placement_positions.png` | Scatter of final (x, y) in **world** coordinates, labeled by rock_id. |

---

## Code layout (this repo)

| File | Role |
|------|------|
| `main_pack.py` | Entry point: argparse, load → pack → export → plot. |
| `rock_loader.py` | Read CSV; parse profiles; rasterize to binary images (with `pixel_scale` / `max_pixel_dim` / `target_max_dim`). |
| `run_packing.py` | Call StablePacking2D (three-pass: strict → strict on unplaced → relaxed); optional MOSEK patch; accumulate placements from all passes; expose `weight_height` and `sequence_order`. |
| `export_placements.py` | Build placement list from result; pixel → world conversion (`placements_pixel_to_world`); write CSV. |
| `plot_packing.py` | `plot_raw_profiles`, `plot_input_profiles`, `plot_wall`, `plot_wall_segmented`, `plot_placements`. |
| `StablePacking2D-master/` | External packing algorithm (submodule or copy). Not modified; we set `INTERLOCKING_PIXEL` and pass `weight_height`. |
| `RockStackingPython.py` | Rhino/Grasshopper script; same packing logic, different I/O (curves in, curves out). This README and the CLI pipeline mirror its parameters where relevant. |

---

## Dimensions and scale (Rhino alignment)

- **Wall:** `wall_height` and `wall_width` are in **pixels**, as in `RockStackingPython.py` (e.g. 200×175).
- **pixel_scale:** Same meaning as in Rhino:
  - **Input:** 1 world unit → `pixel_scale` pixels when rasterizing profiles (stones sized in the same scale as the wall).
  - **Output:** `x_world = x_pixel / pixel_scale`, `y_world = (wall_height - y_pixel) / pixel_scale` (y-up).
- **weight_height:** Passed through to the packer’s placement objective (same as Rhino). Default here is 0.5 for better spread; use 0.03 to match typical Rhino behaviour.

---

## Spread vs clustering

- **Clustering:** Low `--weight-height` (e.g. 0.03) and/or `--sequence random` → stones tend to stack in one region.
- **Spread:** Higher `--weight-height` (0.5–1.0) and `--sequence largest_first` (default) → prefer ground-level placement and large-stones-first order so stones spread along the wall. This is the default in the CLI.

---

## Changelog (summary of changes)

- **Pipeline:** Load rocks from CSV → axis-align stones (like Rhino) → three-pass packing (strict → strict on unplaced → relaxed) → export CSV (world coords) and plots.
- **Args:** `--wall-height`, `--wall-width`, `--pixel-scale`, `--weight-height`, `--sequence`, `--seed`, paths, `--no-plots`.
- **Scale:** Rasterization uses `pixel_scale` (with cap to fit wall); output x,y converted to world via `pixel_scale` and `wall_height`.
- **Spread:** Default `weight_height=0.5`, `sequence=largest_first` to reduce clustering.
- **Plots:** Added `00_raw_profiles.png` for input validation; placement plot uses world coordinates.
- **MOSEK:** If the license is missing, stability check is bypassed so the script still produces placements.
