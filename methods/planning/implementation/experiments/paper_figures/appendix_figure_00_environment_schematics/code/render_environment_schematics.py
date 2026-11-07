#!/usr/bin/env python3
"""Render appendix schematics for the finite-MDP environments."""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = EXPERIMENTS_ROOT.parent

from plot_style import (  # noqa: E402
    apply_conference_matplotlib_style,
    palette_accessibility_report,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PACKAGE_ROOT / "figure_spec.json"
PACKAGE_FIGURES = PACKAGE_ROOT / "figures"
LATEX_FIGURES = REPO_ROOT / "figures"

CELL_FILL = "#FFFFFF"
GRID_EDGE = "#CDCDCD"
TEXT = "#242424"
START = "#E7E7E7"
GOAL = "#DDECCF"
HAZARD = "#EEC7C3"
HOLE = "#CCE4F4"
TRAP = "#EEDCAB"
WALL = "#383838"
MIDLINE = "#5A5A5A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def coord(value: list[int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def coords(values: list[list[int]]) -> list[tuple[int, int]]:
    return [coord(value) for value in values]


def cell_xy(n_rows: int, row: int, col: int) -> tuple[float, float]:
    return float(col), float(n_rows - row - 1)


def draw_grid(
    ax: Any,
    *,
    n_rows: int,
    n_cols: int,
    title: str,
    fills: dict[tuple[int, int], str],
    labels: dict[tuple[int, int], str],
    label_size: float,
) -> None:
    for row in range(n_rows):
        for col in range(n_cols):
            xy = cell_xy(n_rows, row, col)
            ax.add_patch(
                Rectangle(
                    xy,
                    1.0,
                    1.0,
                    facecolor=fills.get((row, col), CELL_FILL),
                    edgecolor=GRID_EDGE,
                    linewidth=0.36,
                )
            )
            label = labels.get((row, col))
            if label:
                ax.text(
                    xy[0] + 0.5,
                    xy[1] + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    color=TEXT,
                    fontweight="bold",
                )
    ax.set_xlim(0.0, n_cols)
    ax.set_ylim(0.0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.35, pad=3.0, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.64)
        spine.set_color(MIDLINE)


def add_row_tag(ax: Any, *, n_rows: int, n_cols: int, row: int, text: str) -> None:
    ax.text(
        n_cols + 0.08,
        n_rows - row - 0.5,
        text,
        ha="left",
        va="center",
        fontsize=6.1,
        color=MIDLINE,
        clip_on=False,
    )


def add_wall(ax: Any, p0: tuple[float, float], p1: tuple[float, float], *, linewidth: float) -> None:
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=WALL,
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=8,
    )


def draw_cliffwalk(ax: Any, env: dict[str, Any]) -> None:
    n_rows, n_cols = [int(value) for value in env["size"]]
    start = coord(env["start"])
    goal = coord(env["goal"])
    fills: dict[tuple[int, int], str] = {start: START, goal: GOAL}
    labels: dict[tuple[int, int], str] = {start: "S", goal: "G"}
    for row_record in env["hazard_rows"]:
        row = int(row_record["row"])
        for col in env["hazard_cols"]:
            fills[(row, col)] = HAZARD
            labels[(row, col)] = "H"
    draw_grid(
        ax,
        n_rows=n_rows,
        n_cols=n_cols,
        title=env["panel_title"],
        fills=fills,
        labels=labels,
        label_size=7.4,
    )
    for row_record in env["hazard_rows"]:
        add_row_tag(
            ax,
            n_rows=n_rows,
            n_cols=n_cols,
            row=int(row_record["row"]),
            text=f"r={int(row_record['reward'])}",
        )


def draw_maze(ax: Any, env: dict[str, Any]) -> None:
    n_rows, n_cols = [int(value) for value in env["size"]]
    start = coord(env["start"])
    goal = coord(env["goal"])
    fills = {start: START, goal: GOAL}
    labels = {start: "S", goal: "G"}
    draw_grid(
        ax,
        n_rows=n_rows,
        n_cols=n_cols,
        title=env["panel_title"],
        fills=fills,
        labels=labels,
        label_size=8.4,
    )
    walls = [set(pair) for pair in env["blocked_adjacencies_state_pairs"]]
    for wall in walls:
        left, right = sorted(wall)
        r0, c0 = divmod(left, n_cols)
        r1, c1 = divmod(right, n_cols)
        if r0 == r1:
            x = max(c0, c1)
            y0 = n_rows - r0 - 1
            add_wall(ax, (x, y0), (x, y0 + 1), linewidth=2.70)
        elif c0 == c1:
            y = n_rows - max(r0, r1)
            add_wall(ax, (c0, y), (c0 + 1, y), linewidth=2.70)


def draw_frozen_lake(ax: Any, env: dict[str, Any]) -> None:
    size = int(env["size"][0])
    holes = coords(env["holes"])
    start = coord(env["start"])
    goal = coord(env["goal"])
    fills = {start: START, goal: GOAL}
    labels = {start: "S", goal: "G"}
    for hole in holes:
        fills[hole] = HOLE
        labels[hole] = "H"
    draw_grid(
        ax,
        n_rows=size,
        n_cols=size,
        title=env["panel_title"],
        fills=fills,
        labels=labels,
        label_size=7.2,
    )


def draw_four_rooms(ax: Any, env: dict[str, Any]) -> None:
    size = int(env["size"][0])
    mid = int(env["wall_midpoint"])
    door_v = int(env["vertical_wall_door_row"])
    door_h = int(env["horizontal_wall_door_col"])
    traps = coords(env["traps"])
    start = coord(env["start"])
    goal = coord(env["goal"])
    fills = {start: START, goal: GOAL}
    labels = {start: "S", goal: "G"}
    for trap in traps:
        fills[trap] = TRAP
        labels[trap] = "T"
    draw_grid(
        ax,
        n_rows=size,
        n_cols=size,
        title=env["panel_title"],
        fills=fills,
        labels=labels,
        label_size=7.0,
    )
    for row in range(size):
        if row == door_v:
            continue
        y0 = size - row - 1
        add_wall(ax, (mid, y0), (mid, y0 + 1), linewidth=1.42)
    y = size - mid
    for col in range(size):
        if col == door_h:
            continue
        add_wall(ax, (col, y), (col + 1, y), linewidth=1.42)


def render() -> dict[str, Any]:
    spec = load_spec()
    environments = spec["environments"]
    apply_conference_matplotlib_style(plt, font_scale=0.90)
    fig = plt.figure(figsize=(5.50, 4.55), constrained_layout=False)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    top_axes = [
        fig.add_axes([0.040, 0.655, 0.264, 0.319]),
        fig.add_axes([0.368, 0.655, 0.264, 0.319]),
        fig.add_axes([0.696, 0.655, 0.264, 0.319]),
    ]
    bottom_axis = fig.add_axes([0.259, 0.035, 0.482, 0.582])

    draw_cliffwalk(top_axes[0], environments["cliffwalk"])
    draw_maze(top_axes[1], environments["maze"])
    draw_frozen_lake(top_axes[2], environments["frozen_lake"])
    draw_four_rooms(bottom_axis, environments["four_rooms"])

    package_pdf = PACKAGE_FIGURES / "environment_schematics.pdf"
    package_png = PACKAGE_FIGURES / "environment_schematics.png"
    latex_pdf = LATEX_FIGURES / "environment_schematics.pdf"
    latex_png = LATEX_FIGURES / "environment_schematics.png"
    PACKAGE_FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    metadata = {
        "Title": "Finite-MDP environment schematics",
        "Creator": "appendix_figure_00_environment_schematics/code/render_environment_schematics.py",
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="There are no gridspecs with layoutgrids.*",
            category=UserWarning,
        )
        fig.savefig(package_pdf, metadata=metadata)
        fig.savefig(package_png, dpi=360)
    plt.close(fig)
    shutil.copy2(package_pdf, latex_pdf)
    shutil.copy2(package_png, latex_png)

    semantic_palette = [START, GOAL, HAZARD, HOLE, TRAP, WALL, MIDLINE]
    source_file_hashes = []
    for source_file in spec["source_files"]:
        source_path = Path(source_file["path"])
        source_record = dict(source_file)
        source_record["sha256"] = sha256(source_path) if source_path.is_file() else None
        source_file_hashes.append(source_record)
    record = {
        "figure": "environment_schematics",
        "renderer": str(Path(__file__).relative_to(REPO_ROOT)),
        "figure_spec": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "outputs": {
            "package_pdf": str(package_pdf.relative_to(REPO_ROOT)),
            "package_png": str(package_png.relative_to(REPO_ROOT)),
            "latex_pdf": str(latex_pdf.relative_to(REPO_ROOT)),
            "latex_png": str(latex_png.relative_to(REPO_ROOT)),
        },
        "sha256": {
            "package_pdf": sha256(package_pdf),
            "package_png": sha256(package_png),
            "latex_pdf": sha256(latex_pdf),
            "latex_png": sha256(latex_png),
        },
        "source_facts": spec["environments"],
        "source_files": source_file_hashes,
        "empty_cell_fill": CELL_FILL,
        "palette_accessibility": palette_accessibility_report(
            semantic_palette,
            min_contrast_vs_white=1.15,
            min_rgb_distance=25.0,
        ),
    }
    metadata_path = PACKAGE_FIGURES / "environment_schematics.metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(metadata_path, LATEX_FIGURES / "environment_schematics.metadata.json")
    manifest = {
        "package_name": spec["package_name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "Source-owned appendix schematic figure for finite-MDP environment layouts.",
        "manuscript_source": spec["manuscript_source"],
        "files": [
            {
                "role": "README",
                "path": str((PACKAGE_ROOT / "README.md").relative_to(REPO_ROOT)),
                "sha256": sha256(PACKAGE_ROOT / "README.md"),
            },
            {
                "role": "figure specification",
                "path": str(SPEC_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256(SPEC_PATH),
            },
            {
                "role": "renderer",
                "path": str(Path(__file__).relative_to(REPO_ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
            {
                "role": "plot style helper",
                "path": str((PACKAGE_ROOT / "code" / "plot_style.py").relative_to(REPO_ROOT)),
                "sha256": sha256(PACKAGE_ROOT / "code" / "plot_style.py"),
            },
            {
                "role": "package PDF",
                "path": str(package_pdf.relative_to(REPO_ROOT)),
                "sha256": sha256(package_pdf),
            },
            {
                "role": "package PNG",
                "path": str(package_png.relative_to(REPO_ROOT)),
                "sha256": sha256(package_png),
            },
            {
                "role": "package metadata",
                "path": str(metadata_path.relative_to(REPO_ROOT)),
                "sha256": sha256(metadata_path),
            },
            {
                "role": "LaTeX-facing PDF",
                "path": str(latex_pdf.relative_to(REPO_ROOT)),
                "sha256": sha256(latex_pdf),
            },
            {
                "role": "LaTeX-facing PNG",
                "path": str(latex_png.relative_to(REPO_ROOT)),
                "sha256": sha256(latex_png),
            },
            {
                "role": "LaTeX-facing metadata",
                "path": "figures/environment_schematics.metadata.json",
                "sha256": sha256(LATEX_FIGURES / "environment_schematics.metadata.json"),
            },
        ],
        "verification_notes": [
            "Environment geometry is read from figure_spec.json.",
            "figure_spec.json records the constructor paths and line ranges used for code comparison.",
            "The Four Rooms panel omits the seed in visible text; the seed remains in provenance because it fixes doors and traps."
        ],
    }
    manifest_path = PACKAGE_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    record = render()
    print(json.dumps(record["outputs"], indent=2))


if __name__ == "__main__":
    main()
