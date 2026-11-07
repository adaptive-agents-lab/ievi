"""Algorithm resolution for maintained hyper-batch runs."""

from __future__ import annotations

from typing import Iterable
import jax.numpy as jnp

from ...core.config import canonical_algorithm_name
from .specs import HyperBatchConfig


def resolve_jax_algorithm_class(name: str):
    """Resolve retained stochastic baseline classes for the JAX backend."""
    from besp.domains.jax_rl import (
        JaxBatchedAltISPDyna,
        JaxBatchedGenAltISPDyna,
        JaxBatchedArchISPDyna,
        JaxBatchedNRRMISPDyna,
        JaxBatchedNR1RMISPDyna,
        JaxBatchedNR1StaircaseWaterfallRMISPDyna,
        JaxBatchedNR1GeoWaterfallRMISPDyna,
        JaxBatchedArchWaterfallISPDyna,
        JaxBatchedDDGeoWaterfallISPDyna,
        JaxBatchedDyna,
        JaxBatchedElasticDDGeoWaterfallISPDyna,
        JaxBatchedISPDyna,
        JaxBatchedOSDyna,
        JaxBatchedQLearning,
        JaxBatchedScheduledWaterfallISPDyna,
        _derive_alt_empirical_mdp_from_state,
        _compute_policy_alt_isp_dyna,
        alt_isp_dyna_step_pure,
        gen_alt_isp_dyna_step_pure,
        _compute_policy_arch_isp_dyna,
        _compute_policy_rm_isp_dyna,
        _compute_policy_nr_rm_isp_dispatch,
        arch_isp_dyna_step_pure,
        nr_rm_isp_dyna_step_pure,
        nr1_rm_isp_dyna_step_pure,
        nr1_staircase_waterfall_rm_isp_dyna_step_pure,
        nr1_geo_waterfall_rm_isp_dyna_step_pure,
        arch_waterfall_isp_dyna_step_pure,
        _compute_policy_from_q,
        _compute_policy_isp_dyna,
        _compute_policy_os_dyna,
        dd_geo_waterfall_isp_dyna_batch_step_pure,
        dyna_step_pure,
        elastic_dd_geo_waterfall_isp_dyna_batch_step_pure,
        isp_dyna_step_pure,
        os_dyna_step_pure,
        q_learning_step_pure,
        scheduled_waterfall_isp_dyna_step_pure,
    )

    lowered = canonical_algorithm_name(name)
    if lowered in {"qlearning", "q_learning"}:
        return (
            JaxBatchedQLearning,
            q_learning_step_pure,
            lambda st, p: _compute_policy_from_q(st.Q),
        )
    if "os" in lowered and "dyna" in lowered:
        return (
            JaxBatchedOSDyna,
            os_dyna_step_pure,
            lambda st, p: _compute_policy_os_dyna(st.r_k, st.P_hat, st.V, p.gamma),
        )
    if lowered == "alt_isp_dyna":
        return (
            JaxBatchedAltISPDyna,
            alt_isp_dyna_step_pure,
            lambda st, p: _compute_policy_alt_isp_dyna(
                _derive_alt_empirical_mdp_from_state(st)[1],
                _derive_alt_empirical_mdp_from_state(st)[0],
                st.V,
                p.gamma,
            ),
        )
    if lowered == "gen_alt_isp_dyna":
        return (
            JaxBatchedGenAltISPDyna,
            gen_alt_isp_dyna_step_pure,
            lambda st, p: _compute_policy_alt_isp_dyna(
                _derive_alt_empirical_mdp_from_state(st)[1],
                _derive_alt_empirical_mdp_from_state(st)[0],
                st.V,
                p.gamma,
            ),
        )
    if lowered == "arch_isp_dyna":
        return (
            JaxBatchedArchISPDyna,
            arch_isp_dyna_step_pure,
            lambda st, p: _compute_policy_arch_isp_dyna(st.r_k, st.P_hat, st.V, p.gamma, p.h),
        )
    if lowered == "nr_rm_isp_dyna":
        return (
            JaxBatchedNRRMISPDyna,
            nr_rm_isp_dyna_step_pure,
            lambda st, p: _compute_policy_nr_rm_isp_dispatch(
                st.defect_r if st.defect_r is not None else st.r_k,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h if st.waterfall_h is not None else p.h,
                p.nr_mode,
            ),
        )
    if lowered == "nr1_rm_isp_dyna":
        return (
            JaxBatchedNR1RMISPDyna,
            nr1_rm_isp_dyna_step_pure,
            lambda st, p: _compute_policy_nr_rm_isp_dispatch(
                st.defect_r if st.defect_r is not None else st.r_k,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h if st.waterfall_h is not None else p.h,
                p.nr_mode,
            ),
        )
    if lowered == "nr1_staircase_waterfall_rm_isp_dyna":
        return (
            JaxBatchedNR1StaircaseWaterfallRMISPDyna,
            nr1_staircase_waterfall_rm_isp_dyna_step_pure,
            lambda st, p: _compute_policy_nr_rm_isp_dispatch(
                st.defect_r if st.defect_r is not None else st.r_k,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h if st.waterfall_h is not None else p.h,
                p.nr_mode,
            ),
        )
    if lowered == "nr1_geo_waterfall_rm_isp_dyna":
        return (
            JaxBatchedNR1GeoWaterfallRMISPDyna,
            nr1_geo_waterfall_rm_isp_dyna_step_pure,
            lambda st, p: _compute_policy_nr_rm_isp_dispatch(
                st.defect_r if st.defect_r is not None else st.r_k,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h if st.waterfall_h is not None else p.h,
                p.nr_mode,
            ),
        )
    if lowered == "dd_geo_waterfall_isp_dyna":
        return (
            JaxBatchedDDGeoWaterfallISPDyna,
            dd_geo_waterfall_isp_dyna_batch_step_pure,
            lambda st, p: st.policy_cache,
        )
    if lowered == "elastic_dd_geo_waterfall_isp_dyna":
        return (
            JaxBatchedElasticDDGeoWaterfallISPDyna,
            elastic_dd_geo_waterfall_isp_dyna_batch_step_pure,
            lambda st, p: st.policy_cache,
        )
    if lowered == "arch_waterfall_isp_dyna":
        return (
            JaxBatchedArchWaterfallISPDyna,
            arch_waterfall_isp_dyna_step_pure,
            lambda st, p: _compute_policy_arch_isp_dyna(
                st.r_k,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h,
            ),
        )
    if lowered == "scheduled_waterfall_isp_dyna":
        return (
            JaxBatchedScheduledWaterfallISPDyna,
            scheduled_waterfall_isp_dyna_step_pure,
            lambda st, p: _compute_policy_isp_dyna(
                st.bar_r,
                st.P_hat,
                st.V,
                p.gamma,
                st.waterfall_h,
            ),
        )
    if "isp" in lowered and "dyna" in lowered:
        return (
            JaxBatchedISPDyna,
            isp_dyna_step_pure,
            lambda st, p: _compute_policy_isp_dyna(st.bar_r, st.P_hat, st.V, p.gamma, p.h),
        )
    if "dyna" in lowered:
        return (
            JaxBatchedDyna,
            dyna_step_pure,
            lambda st, p: _compute_policy_from_q(st.Q),
        )

    raise ValueError(f"Unknown JAX algorithm: {name}")


def group_configs_by_algorithm(
    configs: Iterable[HyperBatchConfig],
    algorithms: Iterable[str],
) -> dict[str, list[HyperBatchConfig]]:
    grouped = {algo: [] for algo in algorithms}
    for cfg in configs:
        if cfg.algorithm in grouped:
            grouped[cfg.algorithm].append(cfg)
    return grouped
