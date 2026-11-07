"""Export the closed-form data shown by the compact-inset Figure 1.

The plot script (`plot_strict_swap_inset.py`) renders the figure but only
emits a small `*.metadata.json` summary. This helper re-runs the same
solvers and writes a richer `*.raw_data.json` that captures every numeric
trajectory the plot uses, plus the illustrative display overrides applied
inside the inset.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLOT_SOURCE = (
    PACKAGE_ROOT
    / "code"
    / "experiments"
    / "two-state-bellman-flow-stiffness"
    / "plot_strict_swap_inset.py"
)
SLUG = "two_state_compact_inset"
OUTPUT = PACKAGE_ROOT / "data" / f"{SLUG}.raw_data.json"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_strict_swap_inset", PLOT_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plot source at {PLOT_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def first_threshold_crossing(queries, values, threshold: float):
    for query, value in zip(queries, values):
        if float(value) <= threshold:
            return float(query)
    return None


def main() -> None:
    module = load_plot_module()
    gamma = 0.99
    a = 0.6
    b = 0.0
    m = 3
    h = 1.0 / (1.0 - gamma)
    outer_steps = 14
    vi_visual_gamma = 0.86
    vi_visual_steps = 22

    problem = module.solve_problem(gamma=gamma, a=a, b=b)
    inner = module.inner_first_step_trace(gamma=gamma, a=a, b=b, h=h, m=m)
    vi_modes_synthetic = module.synthetic_vi_modal_cone(
        gamma_visual=vi_visual_gamma, steps=vi_visual_steps
    )
    exact_modes = module.ievi_outer_trace(
        gamma=gamma, a=a, b=b, h=h, m=1, outer_steps=outer_steps, use_p_hat_eq_p=True
    )
    surrogate_modes = module.ievi_outer_trace(
        gamma=gamma, a=a, b=b, h=h, m=m, outer_steps=outer_steps
    )
    inset_data_bounds = module.modal_zoom_bounds(inner)
    slow_mode_panel = module.slow_mode_panel_data(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=outer_steps,
    )
    threshold = float(slow_mode_panel["threshold"])

    payload = {
        "figure_number": 1,
        "figure_slug": SLUG,
        "parameters": {
            "gamma": gamma,
            "delta": 1.0 - gamma,
            "alpha_self_loop_state1": a,
            "alpha_self_loop_state2": b,
            "surrogate_inner_iterations_m": m,
            "outer_h_mode": "delta-inv",
            "shared_outer_h": h,
            "ie_vi_outer_steps_plotted": outer_steps,
            "vi_cone_synthetic_gamma": vi_visual_gamma,
            "vi_cone_synthetic_steps": vi_visual_steps,
        },
        "model": {
            "reward": clean(problem["reward"]),
            "true_transition": clean(problem["p"]),
            "asymmetric_self_loop_transition": clean(problem["p_hat"]),
            "value_star": clean(problem["value_star"]),
            "initial_error": clean(problem["initial_error"]),
        },
        "main_panel_traces": {
            "vi_synthetic_cone_modes": clean(vi_modes_synthetic),
            "exact_ie_vi_p_hat_eq_p_shared_h_modes": clean(exact_modes),
            "surrogate_ie_vi_a0p6_b0_m3_shared_h_modes": clean(surrogate_modes),
            "vi_y_amplitude_in_panel": 1.0,
            "main_y_scale_applied_in_plot": 1.0,
        },
        "first_step_inner_correction": {
            "anchor_value_u0_actual": clean(inner["values"][0]),
            "inner_values_u0_to_u3": clean(inner["values"]),
            "inner_modes_u0_to_u3": clean(inner["modes"]),
            "exact_implicit_value": clean(inner["exact_value"]),
            "exact_implicit_mode_actual": clean(inner["exact_mode"]),
        },
        "inset_geometry": {
            "axes_fraction_in_main_panel": [0.0548, 0.6318, 0.3274, 0.3023],
            "data_bounds_x_y": list(inset_data_bounds),
            "panel_b_y_limit": module.PANEL_B_Y_LIMIT,
        },
        "inset_illustrative_display_positions": {
            "note": (
                "The U_j marker positions and the diamond position drawn "
                "inside the inset are illustrative overrides, not the actual "
                "algorithm outputs. The algorithm's true positions are in "
                "first_step_inner_correction. The overrides are tuned for "
                "readability: the spiral arrows wrap around the diamond, "
                "U_1 ↔ U_3 are well separated, and every U_j label clears "
                "both its marker and the inset frame at paper scale."
            ),
            "U_1_display_x_y": list(module.PANEL_B_DISPLAY_POS[0]),
            "U_2_display_x_y": list(module.PANEL_B_DISPLAY_POS[1]),
            "U_3_display_x_y": list(module.PANEL_B_DISPLAY_POS[2]),
            "diamond_display_x_y": list(module.PANEL_B_DIAMOND_POS),
            "iterate_marker_size_scatter_s": 8,
            "iterate_marker_size_matches": "main panel surrogate plot markersize=2.8",
        },
        "main_panel_focus_box": {
            "x0_x1_y0_y1": [
                module.FOCUS_BOX["x0"],
                module.FOCUS_BOX["x1"],
                module.FOCUS_BOX["y0"],
                module.FOCUS_BOX["y1"],
            ],
            "centered_on_actual_exact_implicit_mode": True,
        },
        "arrow_curvatures": {
            "U_1_to_U_2_rad": -0.30,
            "U_2_to_U_3_rad": -0.40,
            "rationale": (
                "Negative rad makes each Bezier control point lie on the side "
                "of the chord that is *away* from the central obstacles "
                "(U_3 marker, diamond halo). Positive rad would route both "
                "arcs through their respective obstacles."
            ),
        },
        "slow_mode_panel": {
            "vi_queries": clean(slow_mode_panel["vi_queries"]),
            "vi_slow_mode_error": clean(slow_mode_panel["vi_slow"]),
            "exact_ie_vi_queries": clean(slow_mode_panel["exact_queries"]),
            "exact_ie_vi_slow_mode_error": clean(slow_mode_panel["exact_slow"]),
            "surrogate_ie_vi_queries": clean(slow_mode_panel["surrogate_queries"]),
            "surrogate_ie_vi_slow_mode_error": clean(slow_mode_panel["surrogate_slow"]),
            "threshold": threshold,
            "threshold_crossings": {
                "vi_query_count": int(slow_mode_panel["vi_threshold_query"]),
                "exact_ie_vi_query_count": first_threshold_crossing(
                    slow_mode_panel["exact_queries"],
                    slow_mode_panel["exact_slow"],
                    threshold,
                ),
                "surrogate_ie_vi_query_count": first_threshold_crossing(
                    slow_mode_panel["surrogate_queries"],
                    slow_mode_panel["surrogate_slow"],
                    threshold,
                ),
            },
            "query_counting_convention": (
                "VI uses one true-kernel application per displayed step. "
                "Exact IE-VI is shown as the ideal implicit target path at "
                "one outer query per step. Surrogate IE-VI charges m true-model "
                "defect refreshes per outer update, so its x-axis location is "
                "q = m k."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT.relative_to(PACKAGE_ROOT))


if __name__ == "__main__":
    main()
