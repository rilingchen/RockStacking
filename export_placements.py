"""
Export packing result to a CSV with rock_id, x, y (and optional angle, placed).
"""
from pathlib import Path
import csv
import numpy as np
from skimage.measure import regionprops

# Stone index matrix uses values: 0=empty, 1..n=stone index+1, 253/254/255=boundary/ground
BASE_PIXEL_VALUE = 255
LEFT_BOUND_PIXEL_VALUE = 254
RIGHT_BOUND_PIXEL_VALUE = 253


def placements_from_result(
    result: dict,
    rock_ids: list[int],
) -> list[dict]:
    """
    Build placement rows from packing result. Prefers result["placements_by_stone"]
    (accumulated from all passes); otherwise falls back to stone_index_matrix centroids.

    Returns:
        List of dicts: rock_id, x, y, angle, placed (1 or 0).
    """
    by_stone = {}
    if "placements_by_stone" in result:
        for s, (x, y, angle, placed) in result["placements_by_stone"].items():
            by_stone[s] = {"x": x, "y": y, "angle": angle, "placed": placed}
    else:
        stone_index_matrix = result["stone_index_matrix"]
        mask = (stone_index_matrix > 0) & (stone_index_matrix < RIGHT_BOUND_PIXEL_VALUE)
        labeled = np.where(mask, stone_index_matrix.astype(np.int32), 0)
        for r in regionprops(labeled):
            if r.label >= BASE_PIXEL_VALUE:
                continue
            stone_idx = int(r.label) - 1
            cy, cx = r.centroid
            by_stone[stone_idx] = {
                "x": float(cx),
                "y": float(cy),
                "angle": np.nan,
                "placed": 1,
            }
        sequence = result.get("sequence")
        transformation = result.get("transformation")
        if sequence is not None and transformation is not None and transformation.shape[0] == len(sequence):
            for i, stone_idx in enumerate(sequence):
                if i < transformation.shape[0] and stone_idx in by_stone:
                    by_stone[stone_idx]["angle"] = float(transformation[i, 2])

    out = []
    for s, rid in enumerate(rock_ids):
        if s in by_stone:
            out.append({
                "rock_id": rid,
                "x": by_stone[s]["x"],
                "y": by_stone[s]["y"],
                "angle": by_stone[s]["angle"],
                "placed": by_stone[s]["placed"],
            })
        else:
            out.append({"rock_id": rid, "x": np.nan, "y": np.nan, "angle": np.nan, "placed": 0})
    return out


def placements_pixel_to_world(
    placements: list[dict],
    wall_height_px: int,
    pixel_scale: float,
) -> list[dict]:
    """
    Convert placement (x, y) from pixel coords to world coords.
    Matches RockStackingPython result_to_curves and transforms (center_col, center_row):
      x_world = col / pixel_scale
      y_world = (wall_height_px - row) / pixel_scale   (y up in world)
    """
    out = []
    for row in list(placements):
        r = dict(row)
        if r.get("placed", 0) == 1 and not (np.isnan(r.get("x", np.nan)) or np.isnan(r.get("y", np.nan))):
            r["x"] = float(r["x"]) / pixel_scale
            r["y"] = (wall_height_px - float(r["y"])) / pixel_scale
        out.append(r)
    return out


def export_placements_csv(
    placements: list[dict],
    csv_path: str | Path,
    *,
    include_angle_placed: bool = True,
) -> None:
    """
    Write placements to CSV. Columns: rock_id, x, y [, angle, placed].
    """
    csv_path = Path(csv_path)
    fieldnames = ["rock_id", "x", "y"]
    if include_angle_placed:
        fieldnames += ["angle", "placed"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in placements:
            w.writerow({k: row.get(k) for k in fieldnames})
