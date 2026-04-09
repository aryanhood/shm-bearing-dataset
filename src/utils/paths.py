"""Path helpers for artifacts and reproducible run snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .config import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    models_dir: Path
    reports_dir: Path
    plots_dir: Path
    config_snapshot_path: Path
    metrics_path: Path

    def ensure(self) -> "RunPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class ArtifactPaths:
    models_dir: Path
    plots_dir: Path
    reports_dir: Path
    runs_dir: Path

    @property
    def rf_model_path(self) -> Path:
        return self.models_dir / "rf_model.pkl"

    @property
    def cnn_model_path(self) -> Path:
        return self.models_dir / "cnn_model.pt"

    @property
    def anomaly_model_path(self) -> Path:
        return self.models_dir / "anomaly_model.pkl"

    @property
    def preprocessor_path(self) -> Path:
        return self.models_dir / "preprocessor.pkl"

    @property
    def manifest_path(self) -> Path:
        return self.models_dir / "manifest.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.models_dir / "config_snapshot.yaml"

    def ensure(self) -> "ArtifactPaths":
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        return self

    def create_run_paths(self, purpose: str = "train", run_id: str | None = None) -> RunPaths:
        stamp = run_id or datetime.now().strftime(f"{purpose}-%Y%m%d-%H%M%S")
        root = self.runs_dir / stamp
        return RunPaths(
            root=root,
            models_dir=root / "models",
            reports_dir=root / "reports",
            plots_dir=root / "plots",
            config_snapshot_path=root / "config_snapshot.yaml",
            metrics_path=root / "metrics.json",
        )


def get_artifact_paths(cfg: Mapping | None = None) -> ArtifactPaths:
    config = cfg or get_config()
    artifacts = config["artifacts"]
    paths = ArtifactPaths(
        models_dir=resolve_project_path(artifacts["models_dir"]),
        plots_dir=resolve_project_path(artifacts["plots_dir"]),
        reports_dir=resolve_project_path(artifacts["reports_dir"]),
        runs_dir=resolve_project_path(artifacts.get("runs_dir", "artifacts/runs")),
    )
    return paths
