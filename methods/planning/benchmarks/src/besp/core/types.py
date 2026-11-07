"""Canonical BESP type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeAlias

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy
    from numpy.typing import NDArray

    Array: TypeAlias = NDArray[numpy.floating[Any]] | jax.Array
    IntArray: TypeAlias = NDArray[numpy.integer[Any]] | jax.Array
    BoolArray: TypeAlias = NDArray[numpy.bool_] | jax.Array
else:
    Array = jax.Array
    IntArray = jax.Array
    BoolArray = jax.Array

StateArray = Array
ActionArray = Array
ValueFunction = Array
QFunction = Array
Policy = IntArray
TransitionMatrix = Array
RewardMatrix = Array

ResidualFn = Callable[[ValueFunction], float]
SchedulerFn = Callable[[int], float]
ErrorCurve = Tuple[List[int], List[float]]


def default_metric_mode_for_problem_type(problem_type: str) -> str:
    return "policy_value_error" if str(problem_type) == "control" else "residual_hit_time"


def default_evaluation_pipeline_for_problem_type(problem_type: str) -> str:
    return "train_then_offline" if str(problem_type) == "control" else "online"


def resolve_evaluation_pipeline(
    *,
    problem_type: str,
    metric_mode: str,
    requested_pipeline: str | None = None,
) -> str:
    requested = (
        default_evaluation_pipeline_for_problem_type(problem_type)
        if requested_pipeline is None or str(requested_pipeline).strip() == ""
        else str(requested_pipeline).strip().lower().replace("-", "_")
    )
    if requested not in {"train_then_offline", "online"}:
        raise ValueError("evaluation_pipeline must be one of: train_then_offline, online.")
    if str(problem_type) != "control":
        return "online"
    if str(metric_mode) == "residual_hit_time":
        return "online"
    return requested


@dataclass(frozen=True)
class SamplingContract:
    mode: str = "replay_tape"
    sampling_distribution: str = "uniform_sa"
    replay_manifest_path: str | None = None
    tape_checksum: str | None = None
    seed_values: tuple[int, ...] = ()
    tape_steps: int = 400_000
    consumed_steps: int = 0
    logical_steps: int | None = None
    sample_stride: int = 1
    logical_eval_freq: int | None = None
    schema_version: int = 1

    def as_mapping(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sampling_distribution": self.sampling_distribution,
            "replay_manifest_path": self.replay_manifest_path,
            "tape_checksum": self.tape_checksum,
            "seed_values": list(self.seed_values),
            "tape_steps": self.tape_steps,
            "consumed_steps": self.consumed_steps,
            "logical_steps": self.logical_steps,
            "sample_stride": self.sample_stride,
            "logical_eval_freq": self.logical_eval_freq,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SamplingContract":
        return cls(
            mode=str(mapping.get("mode", "replay_tape")),
            sampling_distribution=str(mapping.get("sampling_distribution", "uniform_sa")),
            replay_manifest_path=(
                None
                if mapping.get("replay_manifest_path") in {None, ""}
                else str(mapping["replay_manifest_path"])
            ),
            tape_checksum=(
                None if mapping.get("tape_checksum") in {None, ""} else str(mapping["tape_checksum"])
            ),
            seed_values=tuple(int(value) for value in mapping.get("seed_values", ())),
            tape_steps=int(mapping.get("tape_steps", 400_000)),
            consumed_steps=int(mapping.get("consumed_steps", 0)),
            logical_steps=(
                None if mapping.get("logical_steps") is None else int(mapping["logical_steps"])
            ),
            sample_stride=int(mapping.get("sample_stride", 1)),
            logical_eval_freq=(
                None
                if mapping.get("logical_eval_freq") is None
                else int(mapping["logical_eval_freq"])
            ),
            schema_version=int(mapping.get("schema_version", 1)),
        )


def normalize_sampling_contract(
    *,
    requested: Mapping[str, Any] | SamplingContract | None,
    seed_values: Sequence[int] | None,
    consumed_steps: int,
    default_tape_steps: int = 400_000,
) -> SamplingContract:
    if requested is None:
        contract = SamplingContract(
            mode="replay_tape",
            sampling_distribution="uniform_sa",
            seed_values=tuple(int(value) for value in (seed_values or ())),
            tape_steps=max(int(default_tape_steps), int(consumed_steps)),
            consumed_steps=int(consumed_steps),
            schema_version=1,
        )
    elif isinstance(requested, SamplingContract):
        contract = requested
    else:
        contract = SamplingContract.from_mapping(requested)
    resolved_seed_values = (
        tuple(int(value) for value in contract.seed_values)
        if contract.seed_values
        else tuple(int(value) for value in (seed_values or ()))
    )
    resolved_consumed_steps = int(contract.consumed_steps or consumed_steps)
    resolved_tape_steps = int(contract.tape_steps or default_tape_steps)
    resolved_sample_stride = max(int(contract.sample_stride or 1), 1)
    resolved_logical_steps = (
        int(contract.logical_steps)
        if contract.logical_steps is not None
        else max(0, resolved_consumed_steps // resolved_sample_stride)
    )
    if resolved_tape_steps < resolved_consumed_steps:
        raise ValueError(
            "sampling_contract.tape_steps must be greater than or equal to consumed_steps; "
            f"received {resolved_tape_steps} and {resolved_consumed_steps}."
        )
    resolved_mode = str(contract.mode).strip().lower().replace("-", "_")
    if resolved_mode not in {"replay_tape", "legacy_prng"}:
        raise ValueError("sampling_contract.mode must be one of: replay_tape, legacy_prng.")
    return SamplingContract(
        mode=resolved_mode,
        sampling_distribution=str(contract.sampling_distribution).strip().lower().replace("-", "_"),
        replay_manifest_path=contract.replay_manifest_path,
        tape_checksum=contract.tape_checksum,
        seed_values=resolved_seed_values,
        tape_steps=resolved_tape_steps,
        consumed_steps=resolved_consumed_steps,
        logical_steps=resolved_logical_steps,
        sample_stride=resolved_sample_stride,
        logical_eval_freq=(
            None if contract.logical_eval_freq is None else int(contract.logical_eval_freq)
        ),
        schema_version=int(contract.schema_version),
    )


@dataclass
class MDP:
    """Tabular Markov Decision Process."""

    P: TransitionMatrix
    R: RewardMatrix
    gamma: float

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    def validate(self) -> bool:
        n_s, n_a, n_s2 = self.P.shape
        r_s, r_a = self.R.shape

        if n_s != n_s2:
            raise ValueError(f"P shape mismatch: {self.P.shape}")
        if n_s != r_s or n_a != r_a:
            raise ValueError(f"R shape {self.R.shape} doesn't match P shape {self.P.shape}")
        if not 0 <= self.gamma < 1:
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")

        row_sums = jnp.sum(jnp.asarray(self.P, dtype=jnp.float64), axis=2)
        if not bool(jnp.allclose(row_sums, 1.0, atol=1e-6)):
            raise ValueError("P rows don't sum to 1")
        return True


@dataclass
class BESPResult:
    queries: List[int]
    errors: List[float]
    V_final: Optional[ValueFunction] = None
    params: Optional[Dict[str, Any]] = None

    def hit_time(self, threshold: float) -> int:
        for q, e in zip(self.queries, self.errors):
            if e <= threshold:
                return q
        return -1


@dataclass
class ControlResult:
    queries: List[int]
    residuals: List[float]
    V_trace: Optional[List[ValueFunction]] = None
    policy: Optional[Policy] = None

    def hit_time(self, threshold: float) -> int:
        for q, r in zip(self.queries, self.residuals):
            if r <= threshold:
                return q
        return -1


@dataclass
class RLResult:
    samples: int
    V_final: ValueFunction
    Q_final: Optional[QFunction] = None
    policy: Optional[Policy] = None
    returns_trace: Optional[List[float]] = None


@dataclass(frozen=True)
class PETrace:
    queries: List[int]
    errors: List[float]
    hit_time: int
    threshold: float
    value_norm: str
    value_normalized: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "queries": list(self.queries),
            "errors": list(self.errors),
            "hit_time": self.hit_time,
            "threshold": self.threshold,
            "value_norm": self.value_norm,
            "value_normalized": self.value_normalized,
        }


@dataclass(frozen=True)
class ControlTrace:
    queries: List[int]
    residuals: List[float]
    value_errors: List[float]
    policy_value_errors: List[float]
    policy_trace: List[List[int]]
    hit_time: int
    threshold: float
    metric_mode: str
    value_norm: str
    value_normalized: bool
    target_id: str | None = None
    target_manifest: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        mapping = {
            "queries": list(self.queries),
            "residuals": list(self.residuals),
            "value_errors": list(self.value_errors),
            "policy_value_errors": list(self.policy_value_errors),
            "policy_trace": [list(policy) for policy in self.policy_trace],
            "hit_time": self.hit_time,
            "threshold": self.threshold,
            "metric_mode": self.metric_mode,
            "value_norm": self.value_norm,
            "value_normalized": self.value_normalized,
        }
        if self.target_id is not None:
            mapping["target_id"] = self.target_id
        if self.target_manifest is not None:
            mapping["target_manifest"] = self.target_manifest
        return mapping


@dataclass(frozen=True)
class OfflineEvalSlabSpec:
    slab_index: int
    capture_start: int
    capture_stop: int
    row_start: int
    row_stop: int

    @property
    def capture_count(self) -> int:
        return self.capture_stop - self.capture_start

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    def as_mapping(self) -> dict[str, Any]:
        return {
            "slab_index": self.slab_index,
            "capture_start": self.capture_start,
            "capture_stop": self.capture_stop,
            "capture_count": self.capture_count,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "row_count": self.row_count,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "OfflineEvalSlabSpec":
        return cls(
            slab_index=int(mapping["slab_index"]),
            capture_start=int(mapping["capture_start"]),
            capture_stop=int(mapping["capture_stop"]),
            row_start=int(mapping["row_start"]),
            row_stop=int(mapping["row_stop"]),
        )


@dataclass(frozen=True)
class OfflineEvalPlan:
    training_steps: int
    env_batch_size: int
    n_states: int
    sample_axis: tuple[int, ...]
    slabs: tuple[OfflineEvalSlabSpec, ...]
    flush_count: int
    policy_dtype: str
    value_dtype: str
    solve_chunk_size: int
    memory_budget_bytes: int

    @property
    def total_capture_steps(self) -> int:
        return len(self.sample_axis)

    @property
    def total_policy_rows(self) -> int:
        return self.total_capture_steps * self.env_batch_size

    def as_mapping(self) -> dict[str, Any]:
        return {
            "training_steps": self.training_steps,
            "env_batch_size": self.env_batch_size,
            "n_states": self.n_states,
            "sample_axis": list(self.sample_axis),
            "total_capture_steps": self.total_capture_steps,
            "total_policy_rows": self.total_policy_rows,
            "flush_count": self.flush_count,
            "policy_dtype": self.policy_dtype,
            "value_dtype": self.value_dtype,
            "solve_chunk_size": self.solve_chunk_size,
            "memory_budget_bytes": self.memory_budget_bytes,
            "slabs": [slab.as_mapping() for slab in self.slabs],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "OfflineEvalPlan":
        return cls(
            training_steps=int(mapping["training_steps"]),
            env_batch_size=int(mapping["env_batch_size"]),
            n_states=int(mapping["n_states"]),
            sample_axis=tuple(int(value) for value in mapping["sample_axis"]),
            slabs=tuple(
                OfflineEvalSlabSpec.from_mapping(slab_mapping)
                for slab_mapping in mapping["slabs"]
            ),
            flush_count=int(mapping["flush_count"]),
            policy_dtype=str(mapping["policy_dtype"]),
            value_dtype=str(mapping["value_dtype"]),
            solve_chunk_size=int(mapping["solve_chunk_size"]),
            memory_budget_bytes=int(mapping["memory_budget_bytes"]),
        )


@dataclass(frozen=True)
class OfflineEvalManifest:
    evaluation_pipeline: str
    algorithm: str
    problem_type: str
    lambda_value: float | None
    seed: int | None
    n_seeds: int
    plan: OfflineEvalPlan
    value_storage_mode: str
    policy_steps_path: str
    evaluated_mask_path: str
    policy_slab_paths: tuple[str, ...]
    value_slab_paths: tuple[str, ...]
    start_state_values_path: str | None
    summary_path: str
    config_labels: tuple[str, ...] = ()
    seed_values: tuple[int, ...] = ()
    config_count: int | None = None
    storage_layout_version: int = 1
    storage_shape: tuple[int, ...] | None = None
    completed_slabs: tuple[int, ...] = ()
    sampling_contract: Mapping[str, Any] | None = None

    @property
    def resolved_config_count(self) -> int:
        if self.config_count is not None:
            return int(self.config_count)
        if self.config_labels:
            return len(self.config_labels)
        if self.n_seeds > 0 and self.plan.env_batch_size % self.n_seeds == 0:
            return self.plan.env_batch_size // self.n_seeds
        return 1

    @property
    def resolved_seed_values(self) -> tuple[int, ...]:
        if self.seed_values:
            return tuple(int(value) for value in self.seed_values)
        return tuple(range(int(self.n_seeds)))

    @property
    def resolved_storage_shape(self) -> tuple[int, ...]:
        if self.storage_shape is not None:
            return tuple(int(value) for value in self.storage_shape)
        if self.storage_layout_version >= 2:
            return (
                self.plan.total_capture_steps,
                self.resolved_config_count,
                int(self.n_seeds),
            )
        return (
            self.plan.total_capture_steps,
            self.plan.env_batch_size,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "evaluation_pipeline": self.evaluation_pipeline,
            "algorithm": self.algorithm,
            "problem_type": self.problem_type,
            "lambda_value": self.lambda_value,
            "seed": self.seed,
            "n_seeds": self.n_seeds,
            "plan": self.plan.as_mapping(),
            "value_storage_mode": self.value_storage_mode,
            "policy_steps_path": self.policy_steps_path,
            "evaluated_mask_path": self.evaluated_mask_path,
            "policy_slab_paths": list(self.policy_slab_paths),
            "value_slab_paths": list(self.value_slab_paths),
            "start_state_values_path": self.start_state_values_path,
            "summary_path": self.summary_path,
            "config_labels": list(self.config_labels),
            "seed_values": list(self.seed_values),
            "config_count": self.config_count,
            "storage_layout_version": self.storage_layout_version,
            "storage_shape": (
                list(self.storage_shape)
                if self.storage_shape is not None
                else None
            ),
            "completed_slabs": list(self.completed_slabs),
            "sampling_contract": (
                dict(self.sampling_contract)
                if self.sampling_contract is not None
                else None
            ),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "OfflineEvalManifest":
        return cls(
            evaluation_pipeline=str(mapping["evaluation_pipeline"]),
            algorithm=str(mapping["algorithm"]),
            problem_type=str(mapping["problem_type"]),
            lambda_value=(
                None if mapping.get("lambda_value") is None else float(mapping["lambda_value"])
            ),
            seed=None if mapping.get("seed") is None else int(mapping["seed"]),
            n_seeds=int(mapping["n_seeds"]),
            plan=OfflineEvalPlan.from_mapping(mapping["plan"]),
            value_storage_mode=str(mapping["value_storage_mode"]),
            policy_steps_path=str(mapping["policy_steps_path"]),
            evaluated_mask_path=str(mapping["evaluated_mask_path"]),
            policy_slab_paths=tuple(str(value) for value in mapping.get("policy_slab_paths", ())),
            value_slab_paths=tuple(str(value) for value in mapping.get("value_slab_paths", ())),
            start_state_values_path=(
                None
                if mapping.get("start_state_values_path") is None
                else str(mapping["start_state_values_path"])
            ),
            summary_path=str(mapping["summary_path"]),
            config_labels=tuple(str(value) for value in mapping.get("config_labels", ())),
            seed_values=tuple(int(value) for value in mapping.get("seed_values", ())),
            config_count=(
                None if mapping.get("config_count") is None else int(mapping["config_count"])
            ),
            storage_layout_version=int(mapping.get("storage_layout_version", 1)),
            storage_shape=(
                None
                if mapping.get("storage_shape") is None
                else tuple(int(value) for value in mapping["storage_shape"])
            ),
            completed_slabs=tuple(int(value) for value in mapping.get("completed_slabs", ())),
            sampling_contract=(
                None
                if mapping.get("sampling_contract") is None
                else dict(mapping["sampling_contract"])
            ),
        )


@dataclass(frozen=True)
class CpuExperimentTask:
    env_name: str
    env_params: Mapping[str, Any]
    lambda_value: float
    seed: int
    max_queries: int
    threshold: float
    algorithms: Mapping[str, Mapping[str, Any]]
    error_mode: str
    problem_type: str
    model_norm: str
    value_norm: str
    value_normalized: bool
    emit_pe_traces: bool = False
    emit_control_traces: bool = False
    metric_mode: str = ""
    evaluation_pipeline: str = ""
    deterministic_target_manifest: str | None = None
    task_metadata: Mapping[str, Any] | None = None
    offline_eval_root: str | None = None

    def __post_init__(self) -> None:
        metric_mode = (
            self.metric_mode
            if self.metric_mode
            else default_metric_mode_for_problem_type(self.problem_type)
        )
        object.__setattr__(self, "metric_mode", metric_mode)
        object.__setattr__(
            self,
            "evaluation_pipeline",
            resolve_evaluation_pipeline(
                problem_type=self.problem_type,
                metric_mode=metric_mode,
                requested_pipeline=self.evaluation_pipeline,
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        mapping = {
            "env_name": self.env_name,
            "env_params": dict(self.env_params),
            "lambda": self.lambda_value,
            "seed": self.seed,
            "max_queries": self.max_queries,
            "threshold": self.threshold,
            "algorithms": dict(self.algorithms),
            "error_mode": self.error_mode,
            "problem_type": self.problem_type,
            "model_norm": self.model_norm,
            "value_norm": self.value_norm,
            "value_normalized": self.value_normalized,
            "emit_pe_traces": self.emit_pe_traces,
            "emit_control_traces": self.emit_control_traces,
            "metric_mode": self.metric_mode,
            "evaluation_pipeline": self.evaluation_pipeline,
        }
        if self.deterministic_target_manifest is not None:
            mapping["deterministic_target_manifest"] = self.deterministic_target_manifest
        if self.task_metadata is not None:
            mapping["task_metadata"] = dict(self.task_metadata)
        if self.offline_eval_root is not None:
            mapping["offline_eval_root"] = self.offline_eval_root
        return mapping


@dataclass(frozen=True)
class GpuBatchTask:
    env_name: str
    env_params: Mapping[str, Any]
    lambda_value: float
    seed: int
    n_seeds: int
    max_queries: int
    threshold: float
    algorithms: Mapping[str, Mapping[str, Any]]
    problem_type: str
    backend: str
    evaluation_pipeline: str = ""
    offline_eval_root: str | None = None
    sampling_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_pipeline",
            resolve_evaluation_pipeline(
                problem_type=self.problem_type,
                metric_mode=default_metric_mode_for_problem_type(self.problem_type),
                requested_pipeline=self.evaluation_pipeline,
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        mapping = {
            "env_name": self.env_name,
            "env_params": dict(self.env_params),
            "lambda": self.lambda_value,
            "seed": self.seed,
            "n_seeds": self.n_seeds,
            "max_queries": self.max_queries,
            "threshold": self.threshold,
            "algorithms": dict(self.algorithms),
            "problem_type": self.problem_type,
            "backend": self.backend,
            "evaluation_pipeline": self.evaluation_pipeline,
        }
        if self.offline_eval_root is not None:
            mapping["offline_eval_root"] = self.offline_eval_root
        if self.sampling_contract is not None:
            mapping["sampling_contract"] = dict(self.sampling_contract)
        return mapping


@dataclass(frozen=True)
class ExperimentExecutionRow:
    lambda_value: float
    seed: int
    results: Mapping[str, int]
    traces: Optional[Mapping[str, PETrace | ControlTrace | Mapping[str, Any]]] = None
    metadata: Mapping[str, Any] | None = None

    def as_mapping(self) -> dict[str, Any]:
        mapping = {
            "lambda": self.lambda_value,
            "seed": self.seed,
            "results": dict(self.results),
        }
        if self.traces:
            mapping["traces"] = dict(self.traces)
        if self.metadata:
            mapping["metadata"] = dict(self.metadata)
        return mapping

    def to_flat_row(self) -> dict[str, Any]:
        row = {"lambda": self.lambda_value, "seed": self.seed}
        row.update(dict(self.results))
        return row


@dataclass(frozen=True)
class HyperBatchTrace:
    mean: Array
    std: Array
    seeds: Array
    x_axis: Array
    diagnostics: Mapping[str, Any] | None = None

    def as_mapping(self) -> dict[str, Any]:
        mapping = {
            "mean": self.mean,
            "std": self.std,
            "seeds": self.seeds,
            "x_axis": self.x_axis,
        }
        if self.diagnostics is not None:
            mapping["diagnostics"] = dict(self.diagnostics)
        return mapping


class Algorithm(Protocol):
    def run(self, **kwargs: Any) -> Any: ...


class BatchedAgent(Protocol):
    V: Array

    def step(self, k: int) -> None: ...

    def get_policy(self) -> IntArray: ...


__all__ = [
    "Algorithm",
    "Array",
    "BESPResult",
    "BatchedAgent",
    "BoolArray",
    "CpuExperimentTask",
    "ControlResult",
    "ControlTrace",
    "ErrorCurve",
    "ExperimentExecutionRow",
    "OfflineEvalManifest",
    "OfflineEvalPlan",
    "OfflineEvalSlabSpec",
    "GpuBatchTask",
    "HyperBatchTrace",
    "IntArray",
    "MDP",
    "PETrace",
    "Policy",
    "QFunction",
    "RLResult",
    "RewardMatrix",
    "SamplingContract",
    "SchedulerFn",
    "StateArray",
    "TransitionMatrix",
    "ValueFunction",
    "default_evaluation_pipeline_for_problem_type",
    "default_metric_mode_for_problem_type",
    "normalize_sampling_contract",
    "resolve_evaluation_pipeline",
]
