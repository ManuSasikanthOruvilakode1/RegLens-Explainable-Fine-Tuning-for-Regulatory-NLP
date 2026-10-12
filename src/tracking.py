"""Optional MLflow experiment tracking.

Runs are logged to a local SQLite store (`mlruns.db`, artifacts in `mlartifacts/`) whenever mlflow
is installed. Set NLP_TRACKING=0 to disable. Any tracking failure is reported once and ignored, so
tracking can never break a training run.

    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db     # browse runs at http://127.0.0.1:5000
"""

import importlib.util
import os
import re
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlruns.db").replace("\\", "/")
ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlartifacts")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def tracking_enabled() -> bool:
    return os.environ.get("NLP_TRACKING", "1") != "0" and importlib.util.find_spec("mlflow") is not None


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flattens nested dicts into dotted keys: {"lora": {"r": 16}} -> {"lora.r": 16}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


class ExperimentTracker:
    """Context manager wrapping one MLflow run; every method is a no-op when tracking is off."""

    def __init__(self, experiment: str, run_name: str, enabled: Optional[bool] = None):
        self.experiment = experiment
        self.run_name = run_name
        self.enabled = tracking_enabled() if enabled is None else enabled
        self._mlflow = None
        self._warned = False

    def _fail(self, exc: Exception) -> None:
        if not self._warned:
            print(f"   [tracking] disabled after error: {exc}")
            self._warned = True
        self.enabled = False

    def __enter__(self) -> "ExperimentTracker":
        if not self.enabled:
            return self
        try:
            import logging
            import mlflow
            logging.getLogger("mlflow").setLevel(logging.ERROR)
            logging.getLogger("alembic").setLevel(logging.ERROR)
            mlflow.set_tracking_uri(TRACKING_URI)
            if mlflow.get_experiment_by_name(self.experiment) is None:
                os.makedirs(ARTIFACT_ROOT, exist_ok=True)
                mlflow.create_experiment(self.experiment, artifact_location="file:///" +
                                         os.path.join(ARTIFACT_ROOT, self.experiment).replace("\\", "/"))
            mlflow.set_experiment(self.experiment)
            mlflow.start_run(run_name=self.run_name)
            self._mlflow = mlflow
        except Exception as exc:  # tracking must never break training
            self._fail(exc)
        return self

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.enabled and self._mlflow:
            try:
                self._mlflow.log_params({k: str(v)[:500] for k, v in flatten(params).items()})
            except Exception as exc:
                self._fail(exc)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._mlflow:
            numeric = {}
            for k, v in flatten(metrics).items():
                if isinstance(v, bool) or v is None:
                    continue
                if isinstance(v, (int, float)):
                    numeric[k] = float(v)
                elif isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                    numeric[f"{k}.low"], numeric[f"{k}.high"] = float(v[0]), float(v[1])
            try:
                self._mlflow.log_metrics({re.sub(r"[^\w\-. /]", "_", k): v for k, v in numeric.items()}, step=step)
            except Exception as exc:
                self._fail(exc)

    def log_artifact(self, path: str) -> None:
        if self.enabled and self._mlflow and os.path.exists(path):
            try:
                self._mlflow.log_artifact(path)
            except Exception as exc:
                self._fail(exc)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mlflow:
            try:
                self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            except Exception:
                pass
