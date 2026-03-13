"""
Main script: load rocks from CSV, run 2D packing, export placements CSV, and plot.
No Rhino dependency. Run from project root with: pipenv run python main_pack.py

Example (spread stones along wall):
  python main_pack.py --wall-height 200 --wall-width 175 --pixel-scale 5
  (defaults: weight-height=0.5, sequence=largest_first for spread; use --weight-height 0.03 for Rhino-style)
"""
import argparse
import sys
from pathlib import Path

# So we can use rotate_axis_align from StablePacking2D (same as RockStackingPython)
_here = Path(__file__).resolve().parent
_stable_src = _here / "StablePacking2D-master" / "src"
if _stable_src.is_dir() and str(_stable_src) not in sys.path:
    sys.path.insert(0, str(_stable_src))

from rock_loader import load_rocks_from_csv
from run_packing import run_packing
from export_placements import (
    placements_from_result,
    placements_pixel_to_world,
    export_placements_csv,
)
from plot_packing import (
    plot_raw_profiles,
    plot_input_profiles,
    plot_wall,
    plot_wall_segmented,
    plot_placements,
)

# Default paths (project root = directory of this file)
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack rocks from CSV into a 2D wall; output placements CSV and plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--wall-height",
        type=int,
        default=200,
        help="Wall height in pixels (same as Rhino wall_height).",
    )
    p.add_argument(
        "--wall-width",
        type=int,
        default=175,
        help="Wall width in pixels (same as Rhino wall_width).",
    )
    p.add_argument(
        "--pixel-scale",
        type=float,
        default=5.0,
        help="Pixels per world unit; used to convert output x,y to world coordinates.",
    )
    p.add_argument(
        "--weight-height",
        type=float,
        default=0.5,
        help="Weight for height: higher (0.5–1.0) = prefer ground, spread stones; low (0.03) = cluster.",
    )
    p.add_argument(
        "--sequence",
        type=str,
        choices=["random", "largest_first"],
        default="largest_first",
        help="Placement order: largest_first = big stones first (spread along ground); random = shuffle.",
    )
    p.add_argument(
        "--target-max-dim",
        type=int,
        default=None,
        help="Longest side of each stone in pixels. Default: min(wall_height, wall_width) // 4.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for placement order.",
    )
    p.add_argument(
        "--data-csv",
        type=Path,
        default=ROOT / "data_rocks.csv",
        help="Input CSV with columns rock_id, profile.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "rock_placements.csv",
        help="Output CSV with rock_id, x, y (world coords), angle, placed.",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "plots",
        help="Directory to save plot images.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plot images.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wall_size = (args.wall_height, args.wall_width)
    # Stone rasterization: match RockStackingPython (curve_to_binary_image uses pixel_scale)
    target_max_dim = args.target_max_dim
    max_pixel_dim = None
    if args.pixel_scale and args.pixel_scale > 0:
        # Rhino-compatible: 1 world unit = pixel_scale pixels; cap so stones fit in wall
        max_pixel_dim = int(min(args.wall_height, args.wall_width) * 0.9)
        max_pixel_dim = max(max_pixel_dim, 10)
        if target_max_dim is None:
            target_max_dim = None  # use pixel_scale only
    elif target_max_dim is None:
        target_max_dim = min(args.wall_height, args.wall_width) // 4
        target_max_dim = max(target_max_dim, 10)

    # 1) Load rocks
    print("Loading rocks from", args.data_csv)
    print(f"  pixel_scale={args.pixel_scale}, wall={args.wall_height}x{args.wall_width} px")
    stones, rock_ids, raw_profiles = load_rocks_from_csv(
        args.data_csv,
        target_max_dim=target_max_dim,
        pixel_scale=args.pixel_scale if args.pixel_scale > 0 else None,
        max_pixel_dim=max_pixel_dim,
    )
    print(f"Loaded {len(stones)} rocks with IDs {rock_ids}")

    if not stones:
        print("No stones to pack. Exiting.")
        return

    # Axis-align each stone (longest axis horizontal) — same as RockStackingPython
    from rotate_stone import rotate_axis_align
    for i in range(len(stones)):
        aligned = rotate_axis_align(stones[i])
        if aligned is not None:
            stones[i] = aligned[0].astype("uint8")

    # 2) Run packing (three-pass: strict -> strict on unplaced -> relaxed)
    print("Running 2D packing (strict → strict → relaxed)...")
    print(f"  wall_size={wall_size} px, weight_height={args.weight_height}, sequence={args.sequence}")
    result = run_packing(
        stones,
        wall_size=wall_size,
        rotation_angle_options=[0, 45, 90, 135, 180, 225, 270, 315],
        seed=args.seed,
        interlocking_pixels=2,
        weight_height=args.weight_height,
        sequence_order=args.sequence,
    )
    print(f"Unplaced: {result['unplaced_stones']}")

    # 3) Placements: pixel coords from result, then convert to world for CSV
    placements_px = placements_from_result(result, rock_ids)
    placements = placements_pixel_to_world(
        placements_px,
        wall_height_px=args.wall_height,
        pixel_scale=args.pixel_scale,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_placements_csv(placements, args.output_csv, include_angle_placed=True)
    print("Wrote", args.output_csv, "(x, y in world units)")

    # 4) Plots (optional)
    if not args.no_plots:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        # Raw CSV outlines (verify input data before rasterization)
        plot_raw_profiles(
            raw_profiles, rock_ids,
            out_path=args.plots_dir / "00_raw_profiles.png",
        )
        print("Saved", args.plots_dir / "00_raw_profiles.png")
        # Rasterized stones (after axis-align) used for packing
        plot_input_profiles(
            stones, rock_ids,
            out_path=args.plots_dir / "01_input_profiles.png",
        )
        plot_wall(result["wall"], out_path=args.plots_dir / "02_wall.png")
        plot_wall_segmented(
            result["stone_index_matrix"],
            out_path=args.plots_dir / "03_wall_by_stone.png",
        )
        # Placement plot in world coords so it matches the CSV
        plot_placements(
            placements,
            out_path=args.plots_dir / "04_placement_positions.png",
            world_coords=True,
        )
        print("Saved plots under", args.plots_dir)


if __name__ == "__main__":
    main()
