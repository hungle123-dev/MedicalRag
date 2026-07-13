from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from medrag_lab.settings import settings


@contextmanager
def tracked_run(name: str, params: dict[str, object]) -> Iterator[object]:
    config = settings()
    Path(config.medrag_artifact_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("medrag-lab")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        yield run


def log_artifact(path: Path) -> None:
    if path.is_file():
        mlflow.log_artifact(str(path))
