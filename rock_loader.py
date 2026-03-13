"""
Load rocks from data_rocks.csv and convert profile polygons to binary images
for StablePacking2D. No Rhino dependency.
"""
from pathlib import Path
import ast
import numpy as np
import cv2


def profile_to_binary_image(
    points: list[list[float]],
    *,
    target_max_dim: int | None = 50,
    pixel_scale: float | None = None,
    max_pixel_dim: int | None = None,
    padding: int = 2,
    dilate_px: int = 2,
) -> np.ndarray:
    """
    Rasterize a 2D profile (list of [x,y] points) into a binary image.
    Image has 1 = rock, 0 = background.

    Scaling (Rhino-compatible when pixel_scale is used):
    - If pixel_scale is set: 1 world unit = pixel_scale pixels (like curve_to_binary_image).
      If max_pixel_dim is set and the image would be larger, scale down so longest side = max_pixel_dim.
    - Else: scale so longest side = target_max_dim pixels.

    Args:
        points: List of [x, y] coordinates (polygon; will be closed if not).
        target_max_dim: Longest side in pixels when pixel_scale is not used.
        pixel_scale: Pixels per world unit (same as Rhino). When set, stone size = world_extent * pixel_scale.
        max_pixel_dim: Cap stone image max dimension (e.g. min(wall_h, wall_w) so stones fit).
        padding: Pixels of padding around the shape.
        dilate_px: Pixels to dilate the shape (helps thin profiles get placed).

    Returns:
        Binary image (uint8): 1 where rock, 0 background.
    """
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        if len(pts) == 2:
            pts = np.vstack([pts, pts[0] + [1e-6, 1e-6], pts[1] + [1e-6, 1e-6]])
        else:
            return np.zeros((1, 1), dtype=np.uint8)

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
        pts = np.vstack([pts, pts[0:1]])

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w = max(x_max - x_min, 1e-10)
    h = max(y_max - y_min, 1e-10)

    if pixel_scale is not None and pixel_scale > 0:
        # Rhino-compatible: 1 world unit = pixel_scale pixels
        scale = float(pixel_scale)
    else:
        scale = (target_max_dim or 50) / max(w, h)

    # Optional cap so stone fits in wall (e.g. max_pixel_dim = min(wall_height, wall_width) - margin)
    img_w_raw = int(np.ceil(w * scale)) + 2 * padding
    img_h_raw = int(np.ceil(h * scale)) + 2 * padding
    if max_pixel_dim is not None and max_pixel_dim > 0:
        longest = max(img_w_raw, img_h_raw)
        if longest > max_pixel_dim:
            scale = (max_pixel_dim - 2 * padding) / max(w, h)
    img_w = int(np.ceil(w * scale)) + 2 * padding
    img_h = int(np.ceil(h * scale)) + 2 * padding
    img_w = max(img_w, 3)
    img_h = max(img_h, 3)

    pts_img = pts.copy()
    pts_img[:, 0] = (pts[:, 0] - x_min) * scale + padding
    pts_img[:, 1] = (y_max - pts[:, 1]) * scale + padding  # flip y for image
    pts_img = pts_img.astype(np.int32)

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts_img], 1)
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        canvas = cv2.dilate(canvas, kernel)
    return canvas


def load_rocks_from_csv(
    csv_path: str | Path,
    *,
    target_max_dim: int | None = 50,
    pixel_scale: float | None = None,
    max_pixel_dim: int | None = None,
    dilate_px: int = 2,
) -> tuple[list[np.ndarray], list[int], list[list[list[float]]]]:
    """
    Load rocks from CSV. Expects columns: rock_id, profile (list of [x,y] points).

    Scaling (match RockStackingPython when using pixel_scale):
    - pixel_scale set: rasterize at pixel_scale pixels per world unit; use max_pixel_dim
      to cap size so stones fit in the wall (e.g. min(wall_height, wall_width) - margin).
    - Else: use target_max_dim so longest side = that many pixels.

    Returns:
        stones: List of binary images (1=rock, 0=background).
        rock_ids: List of rock_id in same order as stones.
        raw_profiles: List of profile point lists (same order), for plotting/validation.
    """
    import csv

    csv_path = Path(csv_path)
    stones = []
    rock_ids = []
    raw_profiles = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["rock_id"])
            profile_str = row["profile"].strip()
            try:
                profile = ast.literal_eval(profile_str)
            except (ValueError, SyntaxError):
                continue
            if not profile or not isinstance(profile, (list, tuple)):
                continue
            img = profile_to_binary_image(
                profile,
                target_max_dim=target_max_dim,
                pixel_scale=pixel_scale,
                max_pixel_dim=max_pixel_dim,
                dilate_px=dilate_px,
            )
            if img.size == 0 or np.sum(img) == 0:
                continue
            stones.append(img)
            rock_ids.append(rid)
            raw_profiles.append([list(pt) for pt in profile])

    return stones, rock_ids, raw_profiles
