"""
Plot packing inputs, wall result, and placement positions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_profiles(
    raw_profiles: list[list[list[float]]],
    rock_ids: list[int],
    out_path: str | Path | None = None,
    *,
    max_per_row: int = 4,
) -> None:
    """
    Plot the raw profile outlines from the CSV (before rasterization).
    Each profile is drawn as a closed polygon in its own subplot so you can
    verify the input data is correct.
    """
    n = len(raw_profiles)
    if n == 0:
        return
    ncols = min(max_per_row, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, (profile, rid) in enumerate(zip(raw_profiles, rock_ids)):
        ax = axes[i]
        pts = np.array(profile)
        if len(pts) < 2:
            ax.set_title(f"Rock {rid} (too few points)")
            ax.set_aspect("equal")
            continue
        # Close the polygon for drawing
        if len(pts) > 2 and (pts[0] != pts[-1]).any():
            pts = np.vstack([pts, pts[0:1]])
        ax.plot(pts[:, 0], pts[:, 1], "b-", linewidth=1.5)
        ax.scatter(pts[:, 0], pts[:, 1], c="b", s=12, zorder=5)
        ax.set_title(f"Rock {rid} (raw CSV outline)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_input_profiles(
    stones: list[np.ndarray],
    rock_ids: list[int],
    out_path: str | Path | None = None,
    *,
    max_per_row: int = 4,
) -> None:
    """Plot each rock profile (binary image) in a grid."""
    n = len(stones)
    if n == 0:
        return
    ncols = min(max_per_row, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, (img, rid) in enumerate(zip(stones, rock_ids)):
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Rock {rid}")
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_wall(
    wall: np.ndarray,
    out_path: str | Path | None = None,
) -> None:
    """Plot the binary wall (1=occupied, 0=empty)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(wall, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Packed wall")
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_wall_segmented(
    stone_index_matrix: np.ndarray,
    out_path: str | Path | None = None,
) -> None:
    """Plot wall with each stone as a different color."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Mask ground/bounds for clearer stone view
    mask = (stone_index_matrix > 0) & (stone_index_matrix < 253)
    vis = np.where(mask, stone_index_matrix, 0)
    ax.imshow(vis, cmap="nipy_spectral")
    ax.set_title("Packed wall (by stone)")
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_placements(
    placements: list[dict],
    out_path: str | Path | None = None,
    *,
    world_coords: bool = False,
) -> None:
    """Scatter plot of (x, y) for placed rocks, labeled by rock_id."""
    placed = [p for p in placements if p.get("placed", 0) == 1]
    if not placed:
        return
    x = [p["x"] for p in placed]
    y = [p["y"] for p in placed]
    ids = [p["rock_id"] for p in placed]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(x, y, alpha=0.8)
    for xi, yi, rid in zip(x, y, ids):
        ax.annotate(str(rid), (xi, yi), fontsize=8, alpha=0.9)
    if world_coords:
        ax.set_xlabel("x (world)")
        ax.set_ylabel("y (world)")
    else:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    ax.set_title("Rock placement positions")
    ax.set_aspect("equal")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
