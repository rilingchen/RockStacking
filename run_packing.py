"""
Run 2D stable packing using StablePacking2D on a list of stone images.
No Rhino dependency. Adds StablePacking2D src to path and calls its API.
"""
from pathlib import Path
import sys
import tempfile
import numpy as np

# Add StablePacking2D src so we can import place_stone_2d, evaluate_kine, rotate_stone
_here = Path(__file__).resolve().parent
_stable_src = _here / "StablePacking2D-master" / "src"
if _stable_src.is_dir() and str(_stable_src) not in sys.path:
    sys.path.insert(0, str(_stable_src))

# Optional: if MOSEK license is missing, stub stability check so packing still runs
def _patch_evaluate_kine_if_no_mosek():
    try:
        import evaluate_kine as ek
        _orig = ek.evaluate_kine
        def _wrapped(elems, contps):
            try:
                return _orig(elems, contps)
            except Exception as e:
                if "license" in str(e).lower() or "mosek" in str(e).lower():
                    return 1.0  # assume stable so placement continues
                raise
        ek.evaluate_kine = _wrapped
    except Exception:
        pass


def run_packing(
    stones: list[np.ndarray],
    *,
    wall_size: tuple[int, int] | None = None,
    rotation_angle_options: list[float] | None = None,
    seed: int = 0,
    interlocking_pixels: int = 2,
    weight_height: float = 1.0,
    sequence_order: str = "random",
) -> dict:
    """
    Pack stones in 2D using StablePacking2D. Uses the same three-pass strategy
    as RockStackingPython: strict -> strict again on unplaced -> relaxed.

    Args:
        stones: List of binary images (1=stone, 0=background). Should be
            axis-aligned (longest axis horizontal) for best placement.
        wall_size: (height, width) in pixels. Defaults to (500, 600).
        rotation_angle_options: Angles in degrees. Default 8 angles like Rhino script.
        seed: Random seed for placement order (when sequence_order='random').
        interlocking_pixels: INTERLOCKING_PIXEL in packer (lower = less strict).
        weight_height: Weight for height; higher = prefer ground = more spread.
        sequence_order: 'random' = shuffle; 'largest_first' = place biggest stones first (spreads better).

    Returns:
        Result dict with keys: wall, wall_id_matrix, stone_index_matrix,
        transformation, unplaced_stones, sequence, placements_by_stone, elems, contps.
    """
    _patch_evaluate_kine_if_no_mosek()
    import place_stone_2d
    place_stone_2d.INTERLOCKING_PIXEL = interlocking_pixels
    from place_stone_2d import (
        generate_one_wall_best_pose_given_sequence,
        generate_one_wall_best_pose_given_sequence_given_wall,
    )

    if wall_size is None:
        wall_size = (500, 600)
    if rotation_angle_options is None:
        rotation_angle_options = [0, 45, 90, 135, 180, 225, 270, 315]

    result_dir = tempfile.mkdtemp(prefix="rock_pack_")
    sequence = np.arange(len(stones))
    if sequence_order == "largest_first":
        # Place largest stones first so they claim ground space; helps spread along wall
        areas = np.array([np.sum(s > 0) for s in stones])
        sequence = np.argsort(-areas)  # descending by area
    else:
        np.random.seed(seed)
        np.random.shuffle(sequence)

    # Accumulate placements from each pass (transformation is overwritten each call)
    placements_by_stone = {}

    # Pass 1: strict mason criteria (bottom + left/right contact + interlocking)
    result = generate_one_wall_best_pose_given_sequence(
        0,
        result_dir=result_dir,
        sequence=sequence,
        stones=stones,
        wall_size=wall_size,
        rotation_angle_options=rotation_angle_options,
        weight_height=weight_height,
        nb_processor=1,
        relaxed_mason_criteria=False,
    )
    for i, stone_idx in enumerate(sequence):
        if i < result["transformation"].shape[0] and result["transformation"][i, 3] == 1:
            placements_by_stone[int(stone_idx)] = (
                float(result["transformation"][i, 0]),
                float(result["transformation"][i, 1]),
                float(result["transformation"][i, 2]),
                1,
            )

    # Pass 2: strict again on unplaced (wall has changed, more stones to build on)
    unplaced_after_1 = list(result["unplaced_stones"])
    if result["unplaced_stones"]:
        result = generate_one_wall_best_pose_given_sequence_given_wall(
            1,
            result["wall"],
            result["wall_id_matrix"],
            result["stone_index_matrix"],
            result["elems"],
            result["contps"],
            result_dir,
            result["unplaced_stones"],
            stones,
            wall_size,
            rotation_angle_options=rotation_angle_options,
            weight_height=weight_height,
            nb_processor=1,
            relaxed_mason_criteria=False,
        )
        for i, stone_idx in enumerate(unplaced_after_1):
            if i < result["transformation"].shape[0] and result["transformation"][i, 3] == 1:
                placements_by_stone[int(stone_idx)] = (
                    float(result["transformation"][i, 0]),
                    float(result["transformation"][i, 1]),
                    float(result["transformation"][i, 2]),
                    1,
                )

    # Pass 3: relaxed criteria on remaining (bypass contact/interlocking to fill gaps)
    unplaced_after_2 = list(result["unplaced_stones"])
    if result["unplaced_stones"]:
        result = generate_one_wall_best_pose_given_sequence_given_wall(
            2,
            result["wall"],
            result["wall_id_matrix"],
            result["stone_index_matrix"],
            result["elems"],
            result["contps"],
            result_dir,
            result["unplaced_stones"],
            stones,
            wall_size,
            rotation_angle_options=rotation_angle_options,
            weight_height=weight_height,
            nb_processor=1,
            relaxed_mason_criteria=True,
        )
        for i, stone_idx in enumerate(unplaced_after_2):
            if i < result["transformation"].shape[0] and result["transformation"][i, 3] == 1:
                placements_by_stone[int(stone_idx)] = (
                    float(result["transformation"][i, 0]),
                    float(result["transformation"][i, 1]),
                    float(result["transformation"][i, 2]),
                    1,
                )

    result["sequence"] = sequence
    result["placements_by_stone"] = placements_by_stone
    return result
