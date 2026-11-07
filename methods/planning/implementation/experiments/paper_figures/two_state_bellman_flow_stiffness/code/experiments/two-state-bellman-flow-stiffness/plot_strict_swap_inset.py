from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
# Package layout: HERE = package/code/experiments/two-state-bellman-flow-stiffness/
# `experiments.common.plot_style` lives at HERE.parents[1]/experiments/common/.
REPO_ROOT = HERE.parents[1]
FIGURE_DIR = HERE / "figures"

sys.path.insert(0, str(REPO_ROOT))
from experiments.common.plot_style import (  # noqa: E402
    apply_conference_matplotlib_style,
    palette_accessibility_report,
)


PALETTE = {
    # Urban Institute palette: grey for VI, blue for exact IE-VI,
    # and orange for surrogate IE-VI.
    "vi": "#9D9D9D",            # Urban mid-grey (VI)
    "ievi_exact": "#1696D2",    # Urban blue (exact IE-VI)
    "ievi_surrogate": "#CA5800", # Urban orange (surrogate IE-VI)
    "axis": "#242424",
    "grid": "#DADADA",
    "shade": "#F7F0EA",
    "zoom_box": "#777777",      # neutral grey for the zoom box (do not
                                 # conflate the zoom region with the surrogate
                                 # trajectory hue).
}

LABEL_BOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.86,
    "pad": 0.65,
}

# Vertical half-extent of the right (zoom) panel. The panel is rendered as a
# symmetric viewport [-PANEL_B_Y_LIMIT, +PANEL_B_Y_LIMIT] around x_-=0 so the
# y=0 reference line sits on the geometric midline. Inside this viewport the
# inner-iterate vertical positions U_1,U_2,U_3 are placed illustratively (see
# panel_b_display_modes); these are inset display coordinates.
PANEL_B_Y_LIMIT = 0.20
# Display positions of the inner iterates and the exact-IE diamond inside
# the symmetric inset viewport. The figure is illustrative (the appendix
# documents this), so positions are tuned for readability rather than
# numerical fidelity. Constraints:
#   - U_1, U_2, U_3 spiral inward toward the diamond.
#   - U_2 sits near the inset's upper-right region.
#   - U_3 sits well clear of the diamond so the U_2 -> U_3 arrowhead is
#     not obscured by the diamond's halo.
#   - The diamond is placed near the inset's horizontal center.
#   - Each marker is far enough from the inset frame that its U_j label
#     clears both the marker and the frame at paper scale.
PANEL_B_DISPLAY_POS = (
    (0.28, -0.078),   # U_1: lower-left, label below, with horizontal
                      #     separation from U_3.
    (0.605, 0.078),   # U_2: upper-right, label above (toward upper-right corner)
    (0.42, -0.07),    # U_3: between U_1 and the diamond, label below
)
PANEL_B_DIAMOND_POS = (0.44, 0.005)


def strict_swap_transition() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)


def asymmetric_self_loop_transition(a: float, b: float) -> np.ndarray:
    """State 1 self-loops with prob a, state 2 self-loops with prob b."""
    return np.asarray([[a, 1.0 - a], [1.0 - b, b]], dtype=float)


def modal_basis() -> np.ndarray:
    return np.asarray([[1.0, 1.0], [1.0, -1.0]])


def modal(value: np.ndarray, value_star: np.ndarray, initial_modal: np.ndarray) -> np.ndarray:
    return np.linalg.solve(modal_basis(), value - value_star) / initial_modal


def solve_problem(*, gamma: float, a: float, b: float) -> dict[str, np.ndarray]:
    p = strict_swap_transition()
    p_hat = asymmetric_self_loop_transition(a, b)
    reward = np.asarray([1.0, 0.0], dtype=float)
    value_star = np.linalg.solve(np.eye(2) - gamma * p, reward)
    initial_error = -value_star
    initial_modal = np.linalg.solve(modal_basis(), initial_error)
    return {
        "p": p,
        "p_hat": p_hat,
        "reward": reward,
        "value_star": value_star,
        "initial_error": initial_error,
        "initial_modal": initial_modal,
    }


def vi_trace(*, gamma: float, steps: int) -> np.ndarray:
    problem = solve_problem(gamma=gamma, a=0.0, b=0.0)
    p = problem["p"]
    r = problem["reward"]
    v_star = problem["value_star"]
    init_modal = problem["initial_modal"]
    v = np.zeros(2)
    modes: list[np.ndarray] = []
    for _ in range(steps + 1):
        modes.append(modal(v, v_star, init_modal))
        v = r + gamma * p @ v
    return np.asarray(modes)


def synthetic_vi_modal_cone(*, gamma_visual: float, steps: int) -> np.ndarray:
    """Synthetic VI modal trajectory at a chosen contraction rate.

    For the strict-swap MDP starting from V=0, VI's modal coordinate at step n
    is (gamma^n, (-1)^n gamma^n). The cone shape (the triangular envelope
    between y = +x and y = -x) is identical for any gamma in (0, 1); gamma
    only controls how many zigzags are needed to fill the cone. The figure
    uses gamma=0.99 for the IE-VI lines (paper-faithful) but a smaller
    gamma_visual for VI so the cone reads as a clean low-frequency backdrop.
    """
    n = np.arange(steps + 1)
    powers = gamma_visual ** n
    signs = (-1.0) ** n
    return np.column_stack([powers, signs * powers])


def ievi_outer_trace(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    outer_steps: int,
    use_p_hat_eq_p: bool = False,
) -> np.ndarray:
    problem = solve_problem(gamma=gamma, a=a, b=b)
    p = problem["p"]
    p_hat = p if use_p_hat_eq_p else problem["p_hat"]
    r = problem["reward"]
    v_star = problem["value_star"]
    init_modal = problem["initial_modal"]
    gamma_h = h * gamma / (1.0 + h)
    solve_matrix = np.eye(2) - gamma_h * p_hat

    v = np.zeros(2)
    modes: list[np.ndarray] = []
    for _ in range(outer_steps + 1):
        modes.append(modal(v, v_star, init_modal))
        anchor = v.copy()
        inner = anchor.copy()
        for _j in range(m):
            defect = r + gamma * (p - p_hat) @ inner
            shifted = anchor / (1.0 + h) + h * defect / (1.0 + h)
            inner = np.linalg.solve(solve_matrix, shifted)
        v = inner
    return np.asarray(modes)


def inner_first_step_trace(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
) -> dict[str, np.ndarray]:
    problem = solve_problem(gamma=gamma, a=a, b=b)
    p = problem["p"]
    p_hat = problem["p_hat"]
    r = problem["reward"]
    v_star = problem["value_star"]
    init_modal = problem["initial_modal"]
    gamma_h = h * gamma / (1.0 + h)
    solve_matrix = np.eye(2) - gamma_h * p_hat

    anchor = np.zeros(2)
    inner = anchor.copy()
    values = [inner.copy()]
    for _j in range(m):
        defect = r + gamma * (p - p_hat) @ inner
        shifted = anchor / (1.0 + h) + h * defect / (1.0 + h)
        inner = np.linalg.solve(solve_matrix, shifted)
        values.append(inner.copy())
    values_arr = np.asarray(values)
    modes_arr = np.asarray([modal(v, v_star, init_modal) for v in values_arr])

    K_true = np.eye(2) + h * (np.eye(2) - gamma * p)
    exact_value = np.linalg.solve(K_true, anchor + h * r)
    exact_mode = modal(exact_value, v_star, init_modal)

    return {
        "values": values_arr,
        "modes": modes_arr,
        "exact_value": exact_value,
        "exact_mode": exact_mode,
        "v_star": v_star,
        "initial_modal": init_modal,
    }


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float,
    alpha: float,
    linestyle: str = "-",
    mutation_scale: float = 8.5,
    connectionstyle: str = "arc3",
    shrink_a: float = 1.5,
    shrink_b: float = 2.5,
    zorder: int = 10,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)


def modal_zoom_bounds(inner: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Symmetric viewport for the right panel.

    The x-extent follows the iterate cluster. The y-extent is fixed
    to the symmetric range [-PANEL_B_Y_LIMIT, +PANEL_B_Y_LIMIT] so the y=0
    reference line crosses the geometric midline of the panel.
    """
    modes = np.asarray(inner["modes"])
    exact_mode = np.asarray(inner["exact_mode"])
    local_x = np.r_[modes[1:, 0], exact_mode[0]]
    x_pad = max(0.04, 0.20 * float(np.ptp(local_x)))
    x0 = float(np.min(local_x) - x_pad)
    x1 = float(np.max(local_x) + x_pad)
    return x0, x1, -PANEL_B_Y_LIMIT, PANEL_B_Y_LIMIT


def panel_a_zoom_box(inner: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Locator rectangle in the left panel matching the right-panel viewport.

    The box delimits the right panel's symmetric vertical viewport so the two
    horizontal connectors between panels are themselves horizontal. The
    x-extent is padded a little wider than the iterate cluster for visual
    balance.
    """
    modes = np.asarray(inner["modes"])
    exact_mode = np.asarray(inner["exact_mode"])
    local_x = np.r_[modes[1:, 0], exact_mode[0]]
    cx = float(exact_mode[0])
    x_half = max(0.10, 0.55 * max(abs(cx - float(np.min(local_x))), abs(float(np.max(local_x)) - cx)))
    return cx - x_half, cx + x_half, -PANEL_B_Y_LIMIT, PANEL_B_Y_LIMIT


def panel_b_display_modes(local_modes: np.ndarray) -> np.ndarray:
    """Display coordinates for U_1,...,U_m inside the inset's viewport.

    The actual algorithm output is preserved in `local_modes`; this helper
    overrides both x- and y-coordinates with PANEL_B_DISPLAY_POS so the
    iterates fit the symmetric viewport with adequate clearance from the
    inset frame, the diamond, and one another. The figure caption flags
    this displacement explicitly.
    """
    display = np.array(local_modes, dtype=float, copy=True)
    for j, (x, y) in enumerate(PANEL_B_DISPLAY_POS[: len(display)]):
        display[j, 0] = x
        display[j, 1] = y
    return display


FOCUS_BOX = {
    # Focus box centered on the exact-IE target at (0.5, 0).
    "x0": 0.435,
    "x1": 0.565,
    "y0": -0.10,
    "y1": 0.10,
}

# Unit scaling preserves the modal initial-error amplitude of 1 in each mode.
MAIN_Y_SCALE = 1.0
SLOW_MODE_THRESHOLD = 1e-3


def draw_main_panel(
    ax: plt.Axes,
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    inner: dict[str, np.ndarray],
    surrogate_label: str,
) -> tuple[float, float, float, float]:
    """Draw the main left-side axes; return the focus-box bounds."""
    # Draw the VI illustration using gamma_visual=0.86 and 22 steps.
    vi = synthetic_vi_modal_cone(gamma_visual=0.86, steps=22)
    exact = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=1,
        outer_steps=14,
        use_p_hat_eq_p=True,
    )
    surrogate = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=14,
    )

    # Apply MAIN_Y_SCALE to each main-panel y coordinate. Its current value
    # is 1.0, so this operation preserves the coordinates.
    vi = vi.copy()
    vi[:, 1] *= MAIN_Y_SCALE
    exact = exact.copy()
    exact[:, 1] *= MAIN_Y_SCALE
    surrogate = surrogate.copy()
    surrogate[:, 1] *= MAIN_Y_SCALE
    start_marker_y = MAIN_Y_SCALE

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.axhline(0.0, color=PALETTE["axis"], lw=0.6, alpha=0.42, zorder=1)

    vi_segments = np.stack([vi[:-1], vi[1:]], axis=1)
    start_dist = np.linalg.norm(vi[:-1], axis=1)
    max_dist = max(float(np.max(start_dist)), 1e-9)
    alpha_min, alpha_max, exponent = 0.12, 0.85, 1.3
    vi_alphas = alpha_min + (alpha_max - alpha_min) * (start_dist / max_dist) ** exponent
    vi_colors = [to_rgba(PALETTE["vi"], alpha=float(a)) for a in vi_alphas]
    ax.add_collection(
        LineCollection(vi_segments, colors=vi_colors, linewidth=0.75, zorder=4)
    )
    ax.plot(
        exact[:, 0],
        exact[:, 1],
        color=PALETTE["ievi_exact"],
        lw=1.40,
        alpha=0.95,
        linestyle=(0, (4, 1.7)),
        marker="D",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=7,
    )
    ax.plot(
        surrogate[:, 0],
        surrogate[:, 1],
        color=PALETTE["ievi_surrogate"],
        lw=1.35,
        marker="o",
        markersize=2.8,
        zorder=6,
    )

    ax.scatter([1.0], [start_marker_y], s=30, color="#111111", edgecolor="white", linewidth=0.7, zorder=9)
    ax.annotate(
        r"$V_0$",
        xy=(1.0, start_marker_y),
        xytext=(-7, -2),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=7.0,
        color="#111111",
        zorder=10,
    )

    box_x0, box_x1 = FOCUS_BOX["x0"], FOCUS_BOX["x1"]
    box_y0, box_y1 = FOCUS_BOX["y0"], FOCUS_BOX["y1"]
    edge_rect = Rectangle(
        (box_x0, box_y0),
        box_x1 - box_x0,
        box_y1 - box_y0,
        facecolor="none",
        edgecolor=PALETTE["zoom_box"],
        alpha=0.75,
        lw=0.7,
        zorder=2,
    )
    ax.add_patch(edge_rect)

    ax.set_xlim(-0.04, 1.04)
    # Modest y-padding so the cone's upper edge does not sit on the spine.
    y_pad = 0.10
    ax.set_ylim(-MAIN_Y_SCALE - y_pad, MAIN_Y_SCALE + y_pad)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels([r"$-1$", "0", "1"])
    ax.set_ylabel(r"fast error mode $x_-$")
    ax.set_xlabel(r"slow error mode $x_+$")
    return box_x0, box_x1, box_y0, box_y1


def draw_inset(
    ax_inset: plt.Axes,
    *,
    inner: dict[str, np.ndarray],
) -> tuple[float, float, float, float]:
    """Draw the floating zoom inset; return its data-coord bounds."""
    modes = np.asarray(inner["modes"])
    exact_mode = np.asarray(inner["exact_mode"])
    local_modes = modes[1:]
    display_modes = panel_b_display_modes(local_modes)
    # Diamond is also displayed at an illustrative position rather than at
    # the algorithm's exact_mode, both to avoid crowding U_3 and to put the
    # diamond near the inset's horizontal centre.
    diamond_display = np.asarray(PANEL_B_DIAMOND_POS, dtype=float)

    # Inset's data viewport: x follows iterate cluster, y is symmetric.
    x0, x1, y0, y1 = modal_zoom_bounds(inner)

    ax_inset.set_facecolor("white")
    # Thin border for the inset; reads as a frame, not a panel.
    for spine in ax_inset.spines.values():
        spine.set_color(PALETTE["zoom_box"])
        spine.set_linewidth(0.6)
        spine.set_alpha(0.75)
    ax_inset.axhline(0.0, color=PALETTE["axis"], lw=0.5, alpha=0.42, zorder=1)

    for idx, (start, end) in enumerate(zip(display_modes[:-1], display_modes[1:])):
        # Negative rad bows each arc *away* from the central obstacles:
        # U_1 -> U_2 bows up-left so it does not cross the U_3 marker;
        # U_2 -> U_3 bows down-right so it does not cross the diamond's
        # halo. (Positive rad would put the arcs through both.)
        rad = -0.30 if idx == 0 else -0.40
        add_arrow(
            ax_inset,
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            color=PALETTE["ievi_surrogate"],
            lw=0.55,
            alpha=0.55,
            mutation_scale=4.5,
            connectionstyle=f"arc3,rad={rad}",
            shrink_a=3.0,
            shrink_b=4.5,
            zorder=9 + idx,
        )

    # Uniform iterate marker size matched to the main panel's surrogate
    # trajectory dots (plot markersize=2.8 ≈ scatter s=8) so the orange dots
    # read consistently across the whole figure.
    ax_inset.scatter(
        display_modes[:, 0],
        display_modes[:, 1],
        s=8,
        color=PALETTE["ievi_surrogate"],
        edgecolor="white",
        linewidth=0.4,
        zorder=12,
    )
    # Diamond + halo: rendered at the illustrative diamond position. The
    # halo masks any arrow tip passing beneath the diamond.
    ax_inset.scatter(
        [diamond_display[0]],
        [diamond_display[1]],
        marker="D",
        s=24,
        facecolor="white",
        edgecolor="white",
        linewidth=0.0,
        alpha=0.85,
        zorder=13,
    )
    ax_inset.scatter(
        [diamond_display[0]],
        [diamond_display[1]],
        marker="D",
        s=12,
        facecolor="white",
        edgecolor=PALETTE["ievi_exact"],
        linewidth=0.85,
        zorder=14,
    )

    # Vertical-outward label placement: each U_j label sits on the side of
    # its marker that points away from the diamond at the cluster centre.
    # The spiral arrows all sweep through the cluster's interior, so
    # pushing the labels to the vertical outside avoids overlapping any
    # arrow. The offset magnitudes are chosen so every label clears the
    # inset's borders at paper scale.
    LABEL_OFFSETS = {
    # Point offsets place U_2 above its marker and U_1/U_3 below theirs.
        1: {"xytext": (0, -9), "ha": "center"},   # below U_1 (lower marker)
        2: {"xytext": (0, 5), "ha": "center"},    # above U_2 (upper marker)
        3: {"xytext": (0, -9), "ha": "center"},   # below U_3 (mid marker)
    }
    for j, point in enumerate(display_modes, start=1):
        spec = LABEL_OFFSETS.get(j, {"xytext": (8, 6), "ha": "left"})
        ax_inset.annotate(
            rf"$U_{j}$",
            xy=(float(point[0]), float(point[1])),
            xytext=spec["xytext"],
            textcoords="offset points",
            fontsize=6.5,
            ha=spec["ha"],
            color=PALETTE["ievi_surrogate"],
            zorder=15,
        )

    ax_inset.set_xlim(x0, x1)
    ax_inset.set_ylim(y0, y1)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    return x0, x1, y0, y1


def value_function_error_norm_from_modes(
    modes: np.ndarray,
    *,
    gamma: float,
    a: float,
    b: float,
) -> np.ndarray:
    """L_inf norm of V - V* recovered from normalized modal coordinates.

    ``modes[i]`` is ``M^{-1} (V_i - V*) / initial_modal``; multiplying by
    ``initial_modal`` and applying the modal basis ``M`` reconstructs
    ``V_i - V*`` in standard coordinates. The L_inf norm of that is what
    panel (b) plots, normalized by its initial value.
    """
    problem = solve_problem(gamma=gamma, a=a, b=b)
    init_modal = problem["initial_modal"]
    M = modal_basis()
    unnormalized_modal = modes * init_modal[None, :]
    standard_errors = unnormalized_modal @ M.T
    return np.max(np.abs(standard_errors), axis=1)


def slow_mode_panel_data(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    outer_steps: int = 14,
) -> dict[str, np.ndarray | int | float]:
    """Compute the right-panel relative-error decay traces.

    The y-axis quantity is the relative L_inf value-function error
    ``||V_k - V*||_inf / ||V_0 - V*||_inf``, which differs from panel
    (a)'s modal x-axis quantity. After IE-VI's first outer step kills
    the fast mode, this norm tracks the slow-mode amplitude up to a
    multiplicative constant; for VI it follows ``gamma^q`` exactly.

    The surrogate curve is plotted against charged true-model defect
    refreshes, so one outer update advances the x-axis by ``m``. The
    exact IE-VI curve is the ideal implicit target path and is shown at
    one unit per outer update as a reference for the step-size mechanism.
    """
    exact_modes = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=1,
        outer_steps=outer_steps,
        use_p_hat_eq_p=True,
    )
    surrogate_modes = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=outer_steps,
    )
    max_query = outer_steps * m
    vi_queries = np.arange(max_query + 1)
    exact_queries = np.arange(outer_steps + 1)
    surrogate_queries = np.arange(outer_steps + 1) * m
    vi_threshold_query = math.ceil(math.log(SLOW_MODE_THRESHOLD) / math.log(gamma))

    exact_norms = value_function_error_norm_from_modes(
        exact_modes, gamma=gamma, a=a, b=b
    )
    surrogate_norms = value_function_error_norm_from_modes(
        surrogate_modes, gamma=gamma, a=a, b=b
    )
    exact_relative = exact_norms / exact_norms[0]
    surrogate_relative = surrogate_norms / surrogate_norms[0]
    # VI's relative L_inf error decays exactly as gamma^q (the slow
    # eigenvector dominates the cone envelope; closed-form, no need to
    # run the recurrence).
    vi_relative = gamma ** vi_queries

    return {
        "vi_queries": vi_queries,
        "vi_relative_error": vi_relative,
        "exact_queries": exact_queries,
        "exact_relative_error": exact_relative,
        "surrogate_queries": surrogate_queries,
        "surrogate_relative_error": surrogate_relative,
        "threshold": SLOW_MODE_THRESHOLD,
        "vi_threshold_query": vi_threshold_query,
        "max_query": max_query,
    }


def draw_slow_mode_panel(
    ax: plt.Axes,
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    outer_steps: int = 14,
) -> dict[str, object]:
    """Draw the right-panel slow-mode convergence comparison.

    Position on a common scale carries the one quantitative statement
    panel (a) cannot make: the query budget needed to drive the slow-mode
    error below a fixed threshold. Series identity is deferred to the
    figure-level shared legend; in-panel ink is reserved for the threshold
    reference and the threshold-crossing drop-lines.
    """
    data = slow_mode_panel_data(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=outer_steps,
    )
    vi_q = np.asarray(data["vi_queries"], dtype=float)
    vi_y = np.asarray(data["vi_relative_error"], dtype=float)
    exact_q = np.asarray(data["exact_queries"], dtype=float)
    exact_y = np.asarray(data["exact_relative_error"], dtype=float)
    surrogate_q = np.asarray(data["surrogate_queries"], dtype=float)
    surrogate_y = np.asarray(data["surrogate_relative_error"], dtype=float)
    threshold = float(data["threshold"])
    max_query = int(data["max_query"])
    vi_threshold_query = int(data["vi_threshold_query"])

    def first_below(q_arr: np.ndarray, y_arr: np.ndarray, thresh: float) -> int:
        mask = y_arr <= thresh
        if not mask.any():
            return -1
        return int(q_arr[int(np.argmax(mask))])

    exact_threshold_query = first_below(exact_q, exact_y, threshold)
    surrogate_threshold_query = first_below(surrogate_q, surrogate_y, threshold)

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.set_yscale("log")
    ax.set_xlim(0, max_query)
    y_min = 4e-5
    ax.set_ylim(y_min, 1.35)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(["0", "10", "20", "30", "40"])
    ax.set_yticks([1.0, 1e-2, 1e-3, 1e-4])
    ax.set_yticklabels([r"$1$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"])
    ax.tick_params(axis="both", which="major", length=2.5, pad=2.0)
    ax.grid(axis="y", which="major", color=PALETTE["grid"], alpha=0.45, linewidth=0.45, zorder=0)

    # Drop-lines from each visible threshold crossing to the x-axis, in
    # the series colour. The horizontal distance from the y-axis to the
    # drop tick is the query-cost --- the one quantitative statement
    # panel (a) cannot make.
    for q_cross, color in [
        (exact_threshold_query, PALETTE["ievi_exact"]),
        (surrogate_threshold_query, PALETTE["ievi_surrogate"]),
    ]:
        if q_cross < 0 or q_cross > max_query:
            continue
        ax.plot(
            [q_cross, q_cross],
            [y_min, threshold],
            color=color,
            lw=0.55,
            linestyle=(0, (1.6, 1.6)),
            alpha=0.65,
            zorder=2,
        )

    # Threshold reference line at y = SLOW_MODE_THRESHOLD, stopped short
    # of the right-edge "->688" annotation so the dashes do not bleed
    # past it. The y-tick at 10^{-3} anchors the line on the y-axis, so
    # no in-panel text label is needed.
    threshold_line_end = max_query - 3.5
    ax.plot(
        [0, threshold_line_end],
        [threshold, threshold],
        color=PALETTE["axis"],
        lw=0.55,
        alpha=0.50,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    ax.plot(
        vi_q,
        vi_y,
        color=PALETTE["vi"],
        lw=1.05,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        exact_q,
        exact_y,
        color=PALETTE["ievi_exact"],
        lw=1.35,
        linestyle=(0, (4, 1.7)),
        marker="D",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        zorder=5,
    )
    ax.plot(
        surrogate_q,
        surrogate_y,
        color=PALETTE["ievi_surrogate"],
        lw=1.35,
        marker="o",
        markersize=2.8,
        zorder=4,
    )

    # VI's threshold crossing is far off-axis at q=688; the right-edge
    # annotation reads "VI: q=688" so the reader sees the series identity
    # and the off-axis crossing in one chunk. The arrow on the value sits
    # in the gap left by the truncated threshold line.
    ax.text(
        max_query - 0.4,
        threshold,
        rf"VI: $q\!=\!{vi_threshold_query}\!\rightarrow$",
        ha="right",
        va="center",
        fontsize=6.5,
        color=PALETTE["vi"],
        zorder=6,
    )

    # Colour the x-tick labels at the visible threshold crossings to tie
    # them to the matching drop-line colour. No extra text is needed: the
    # tick value is already the query-cost.
    label_color_map = {
        exact_threshold_query: PALETTE["ievi_exact"],
        surrogate_threshold_query: PALETTE["ievi_surrogate"],
    }
    for tick_label in ax.get_xticklabels():
        try:
            value = int(tick_label.get_text())
        except ValueError:
            continue
        if value in label_color_map:
            tick_label.set_color(label_color_map[value])

    ax.set_xlabel("true-model queries")
    ax.set_ylabel("relative error")

    return {
        "vi_queries": vi_q.tolist(),
        "vi_relative_error": vi_y.tolist(),
        "exact_queries": exact_q.tolist(),
        "exact_relative_error": exact_y.tolist(),
        "surrogate_queries": surrogate_q.tolist(),
        "surrogate_relative_error": surrogate_y.tolist(),
        "threshold": threshold,
        "vi_threshold_query": vi_threshold_query,
        "exact_threshold_query": exact_threshold_query,
        "surrogate_threshold_query": surrogate_threshold_query,
        "max_query": max_query,
    }


def build_figure(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    output_stem: str,
    dpi: int,
    inset_position: tuple[float, float, float, float] = (0.0548, 0.6318, 0.3274, 0.3023),
    title_suffix: str | None = None,
) -> dict[str, object]:
    inner = inner_first_step_trace(gamma=gamma, a=a, b=b, h=h, m=m)

    apply_conference_matplotlib_style(
        plt,
        font_scale=0.98,
        palette=(PALETTE["vi"], PALETTE["ievi_exact"], PALETTE["ievi_surrogate"]),
    )
    plt.rcParams.update({"text.usetex": False})

    # Figure native width matches the NeurIPS textwidth (5.5 in,
    # neurips_2026.sty); rendering at \linewidth in the manuscript leaves
    # fonts and line weights at their authored point sizes (no scaling).
    # The figure height accommodates a single shared legend
    # below both panels via constrained_layout's "outside lower" loc.
    fig = plt.figure(figsize=(5.5, 2.65), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.07)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    surrogate_label = rf"IE-VI, surrogate ($h={h:g}$, $m={m}$)"

    box_x0, box_x1, box_y0, box_y1 = draw_main_panel(
        ax_main,
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        inner=inner,
        surrogate_label=surrogate_label,
    )

    ax_inset = ax_main.inset_axes(inset_position)
    ix0, ix1, iy0, iy1 = draw_inset(ax_inset, inner=inner)

    slow_mode_panel = draw_slow_mode_panel(
        ax_right,
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=14,
    )

    # Connector lines from the focus-box corners in the main axes to the
    # bottom corners of the inset. The inset sits above the box, so the
    # connectors fan upward.
    for (box_corner_xy, inset_corner_xy) in [
        ((box_x0, box_y1), (ix0, iy0)),
        ((box_x1, box_y1), (ix1, iy0)),
    ]:
        connector = ConnectionPatch(
            xyA=box_corner_xy,
            coordsA=ax_main.transData,
            xyB=inset_corner_xy,
            coordsB=ax_inset.transData,
            color=PALETTE["zoom_box"],
            alpha=0.65,
            lw=0.7,
            zorder=8,
        )
        fig.add_artist(connector)

    # Single shared legend below both panels. Colour, dash, and marker
    # pairs match exactly the styles drawn in panel (a) and panel (b),
    # so neither panel needs its own in-axes legend or floating direct
    # labels.
    legend_handles = [
        Line2D([0], [0], color=PALETTE["vi"], lw=1.0, label="VI"),
        Line2D(
            [0], [0],
            color=PALETTE["ievi_exact"],
            lw=1.2,
            linestyle=(0, (4, 1.7)),
            marker="D",
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=r"IE-VI, $\hat P=P$",
        ),
        Line2D(
            [0], [0],
            color=PALETTE["ievi_surrogate"],
            lw=1.15,
            marker="o",
            markersize=2.4,
            label=rf"IE-VI, surrogate ($h={h:g},\,m={m}$)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=2.0,
        borderaxespad=0.0,
    )

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=8.5, y=1.02)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{output_stem}.png"
    pdf_path = FIGURE_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    p = strict_swap_transition()
    p_hat = asymmetric_self_loop_transition(a, b)
    report = palette_accessibility_report(
        [PALETTE["vi"], PALETTE["ievi_exact"], PALETTE["ievi_surrogate"]]
    )

    summary = {
        "gamma": gamma,
        "delta": 1.0 - gamma,
        "alpha_self_loop_state1": a,
        "alpha_self_loop_state2": b,
        "h": h,
        "h_over_inv_delta": h * (1.0 - gamma),
        "m": m,
        "true_transition": p.tolist(),
        "asymmetric_self_loop_transition": p_hat.tolist(),
        "focus_box": [box_x0, box_x1, box_y0, box_y1],
        "inset_axes_fraction": list(inset_position),
        "inset_data_bounds": [ix0, ix1, iy0, iy1],
        "inner_modes": np.asarray(inner["modes"]).tolist(),
        "exact_implicit_mode": np.asarray(inner["exact_mode"]).tolist(),
        "slow_mode_panel": slow_mode_panel,
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "palette_accessibility_report": report,
    }
    (FIGURE_DIR / f"{output_stem}.metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the strict-swap two-state IE-VI modal path with an asymmetric self-loop surrogate."
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--a", type=float, default=0.6, help="Self-loop probability for state 1 in surrogate")
    parser.add_argument("--b", type=float, default=0.0, help="Self-loop probability for state 2 in surrogate")
    parser.add_argument("--h", type=float, default=None, help="Outer step size; defaults to 1/(1-gamma)")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--output-stem", default="two_state_compact_inset")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h = args.h if args.h is not None else 1.0 / (1.0 - args.gamma)
    summary = build_figure(
        gamma=args.gamma,
        a=args.a,
        b=args.b,
        h=h,
        m=args.m,
        output_stem=args.output_stem,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
