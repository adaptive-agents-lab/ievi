"""Centralized output and log path policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify_name(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip()).strip("._-")
    return slug or "run"


def canonical_slurm_log_dir(
    *,
    launcher: str,
    run_id: str,
    outputs_root: Path = Path("outputs"),
) -> Path:
    return outputs_root / "slurm_logs" / slugify_name(launcher) / slugify_name(run_id)


@dataclass(frozen=True)
class RunArtifacts:
    launcher: str
    run_id: str
    run_dir: Path
    raw_dir: Path
    plots_dir: Path
    logs_dir: Path
    offline_eval_dir: Path
    config_snapshot_path: Path
    results_csv_path: Path
    results_json_path: Path
    summary_json_path: Path
    offline_eval_summary_path: Path

    def plot_path(self, filename: str) -> Path:
        return self.plots_dir / filename

    def raw_path(self, filename: str) -> Path:
        return self.raw_dir / filename

    def log_path(self, filename: str) -> Path:
        return self.logs_dir / filename

    def offline_eval_case_dir(self, *parts: str) -> Path:
        path = self.offline_eval_dir
        for part in parts:
            path = path / slugify_name(part)
        return path

    def ensure_directories(self) -> None:
        for path in (
            self.run_dir,
            self.raw_dir,
            self.plots_dir,
            self.logs_dir,
            self.offline_eval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_run_artifacts(
    *,
    experiment_name: str,
    launcher: str,
    output_dir: str | Path | None = None,
    outputs_root: Path = Path("outputs"),
    category: str = "runs",
    run_id: str | None = None,
) -> RunArtifacts:
    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        run_dir = outputs_root / category / f"{resolved_run_id}_{slugify_name(experiment_name)}"
    else:
        run_dir = Path(output_dir)

    return RunArtifacts(
        launcher=launcher,
        run_id=resolved_run_id,
        run_dir=run_dir,
        raw_dir=run_dir / "raw",
        plots_dir=run_dir / "plots",
        logs_dir=canonical_slurm_log_dir(
            launcher=launcher, run_id=resolved_run_id, outputs_root=outputs_root
        ),
        offline_eval_dir=run_dir / "raw" / "offline_eval",
        config_snapshot_path=run_dir / "config.yaml",
        results_csv_path=run_dir / "results.csv",
        results_json_path=run_dir / "raw" / "results.json",
        summary_json_path=run_dir / "summary.json",
        offline_eval_summary_path=run_dir / "raw" / "offline_eval" / "summary.json",
    )


def archive_root_logs(
    *,
    repo_root: str | Path,
    archive_date: str,
    outputs_root: str | Path | None = None,
) -> list[Path]:
    repo_path = Path(repo_root)
    base_outputs = Path(outputs_root) if outputs_root is not None else repo_path / "outputs"
    archive_dir = base_outputs / "slurm_logs" / "repo-root" / archive_date
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved: list[Path] = []
    for pattern in ("slurm-*.out", "slurm-*.err", "nohup.out"):
        for src in sorted(repo_path.glob(pattern)):
            destination = archive_dir / src.name
            if destination.exists():
                destination.unlink()
            src.rename(destination)
            moved.append(destination)
    return moved
