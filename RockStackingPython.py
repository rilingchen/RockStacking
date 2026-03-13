#! python3
# requirements: opencv-python
# requirements: scikit-image
# requirements: matplotlib
# requirements: trimesh
# requirements: Mosek
import sys
sys.path.append(r"/Users/r2d2/Desktop/RockStacking/StablePacking2D-master/src")

# Override interlocking before importing
import place_stone_2d
place_stone_2d.INTERLOCKING_PIXEL = interlocking_pixels

import numpy as np
import Rhino.Geometry as rg
import tempfile
import cv2
from place_stone_2d import (generate_one_wall_best_pose_given_sequence,
                             generate_one_wall_best_pose_given_sequence_given_wall,
                             base_pixel_value, left_bound_pixel_value, right_bound_pixel_value)
from rotate_stone import rotate_axis_align

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────

def curve_to_binary_image(crv, pixel_scale):
    bbox = crv.GetBoundingBox(True)
    min_pt = bbox.Min
    polyline_curve = crv.ToPolyline(0.1, 5, 0.01, 10000)
    polyline = polyline_curve.ToPolyline()
    pixel_pts = []
    for pt in polyline:
        px = int((pt.X - min_pt.X) * pixel_scale)
        py = int((pt.Y - min_pt.Y) * pixel_scale)
        pixel_pts.append([px, py])
    stone_w = int((bbox.Max.X - bbox.Min.X) * pixel_scale) + 2
    stone_h = int((bbox.Max.Y - bbox.Min.Y) * pixel_scale) + 2
    img = np.zeros((stone_h, stone_w), dtype=np.uint8)
    pts_array = np.array(pixel_pts, dtype=np.int32)
    cv2.fillPoly(img, [pts_array], 1)
    img, _ = rotate_axis_align(img)
    return img.astype('uint8')

def result_to_curves(stone_index_matrix, pixel_scale):
    curves = []
    unique_ids = np.unique(stone_index_matrix)
    h = stone_index_matrix.shape[0]
    boundary_ids = [float(base_pixel_value), float(left_bound_pixel_value), float(right_bound_pixel_value)]
    for sid in unique_ids:
        if sid <= 0:
            continue
        if sid in boundary_ids:
            continue
        mask = np.zeros_like(stone_index_matrix, dtype=np.uint8)
        mask[stone_index_matrix == sid] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]
        pts = []
        for point in contour:
            px, py = point[0]
            x = px / pixel_scale
            y = (h - py) / pixel_scale
            pts.append(rg.Point3d(x, y, 0))
        if len(pts) > 1:
            pts.append(pts[0])
            curves.append(rg.PolylineCurve(pts))
    return curves

# ── MAIN ───────────────────────────────────────────────────────────────────

try:
    if not hasattr(stone_curves, '__iter__'):
        stone_curves = [stone_curves]

    # Convert all Rhino curves to binary images
    stone_images = []
    for crv in stone_curves:
        img = curve_to_binary_image(crv, pixel_scale)
        stone_images.append(img)

    result_dir = tempfile.mkdtemp()
    sequence = np.arange(len(stone_images))

    log = []

    # ── PASS 1: strict mason criteria ─────────────────────────────────────
    result = generate_one_wall_best_pose_given_sequence(
        wall_i=0,
        result_dir=result_dir,
        sequence=sequence,
        stones=stone_images,
        wall_size=(wall_height, wall_width),
        rotation_angle_options=[0, 45, 90, 135, 180, 225, 270, 315],
        weight_height=weight_height,
        nb_processor=1,
        relaxed_mason_criteria=False
    )

    placed = len(sequence) - len(result['unplaced_stones'])
    log.append(f"Pass 1 (strict): Placed {placed}/{len(sequence)}, Unplaced: {len(result['unplaced_stones'])}")

    # ── PASS 2: strict again on unplaced stones ────────────────────────────
    # Sometimes a second strict pass places more stones
    # because the wall has changed since pass 1
    if result['unplaced_stones']:
        prev_unplaced = len(result['unplaced_stones'])
        result = generate_one_wall_best_pose_given_sequence_given_wall(
            1,
            result['wall'],
            result['wall_id_matrix'],
            result['stone_index_matrix'],
            result['elems'],
            result['contps'],
            result_dir,
            result['unplaced_stones'],
            stone_images,
            (wall_height, wall_width),
            rotation_angle_options=[0, 45, 90, 135, 180, 225, 270, 315],
            weight_height=weight_height,
            nb_processor=1,
            relaxed_mason_criteria=False
        )
        newly_placed = prev_unplaced - len(result['unplaced_stones'])
        log.append(f"Pass 2 (strict again): Placed {newly_placed} more, Unplaced: {len(result['unplaced_stones'])}")

    # ── PASS 3: relaxed criteria on remaining unplaced stones ──────────────
    # This bypasses contact/interlocking requirements
    # so remaining stones can fill gaps
    if result['unplaced_stones']:
        prev_unplaced = len(result['unplaced_stones'])
        result = generate_one_wall_best_pose_given_sequence_given_wall(
            2,
            result['wall'],
            result['wall_id_matrix'],
            result['stone_index_matrix'],
            result['elems'],
            result['contps'],
            result_dir,
            result['unplaced_stones'],
            stone_images,
            (wall_height, wall_width),
            rotation_angle_options=[0, 45, 90, 135, 180, 225, 270, 315],
            weight_height=weight_height,
            nb_processor=1,
            relaxed_mason_criteria=True  # bypass contact rules
        )
        newly_placed = prev_unplaced - len(result['unplaced_stones'])
        log.append(f"Pass 3 (relaxed): Placed {newly_placed} more, Unplaced: {len(result['unplaced_stones'])}")

    # ── OUTPUT ─────────────────────────────────────────────────────────────
    a = result_to_curves(result['stone_index_matrix'], pixel_scale)
    b = "\n".join(log)
    b += f"\n\nFinal: {len(a)} stones placed"
    c = np.unique(result['stone_index_matrix']).tolist()

    # Transformation data for mesh orientation
    transforms = []
    h = result['stone_index_matrix'].shape[0]
    for stone_idx in range(len(stone_images)):
        target_id = float(stone_idx + 1)
        pixels = np.argwhere(result['stone_index_matrix'] == target_id)
        if len(pixels) == 0:
            continue
        center_row = float(np.mean(pixels[:, 0]))
        center_col = float(np.mean(pixels[:, 1]))
        target_x = center_col / pixel_scale
        target_y = (h - center_row) / pixel_scale
        angle_deg = 0.0
        for i, seq_idx in enumerate(sequence):
            if seq_idx == stone_idx:
                # Check index is within bounds of transformation matrix
                if i < result['transformation'].shape[0]:
                    angle_deg = float(result['transformation'][i][2])
                break
        transforms.append(f"{stone_idx},{target_x},{target_y},{angle_deg}")
    d = transforms

except Exception as e:
    import traceback
    a = []
    b = f"Error: {traceback.format_exc()}"
    c = []
    d = []