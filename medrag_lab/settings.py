from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_base_url: str = ""
    gateway_generator_model: str = ""
    gateway_judge_gemini: str = ""
    gateway_judge_openai: str = ""
    gateway_judge_qwen_or_deepseek: str = ""
    gateway_judge_model: str = ""  # compatibility with the current local .env only

    medrag_data_dir: Path = ROOT / "data" / "raw" / "bioasq"
    medrag_index_dir: Path = ROOT / "artifacts" / "indexes"
    medrag_artifact_dir: Path = ROOT / "artifacts" / "runs"
    mlflow_tracking_uri: str = f"sqlite:///{(ROOT / 'artifacts' / 'mlflow.db').as_posix()}"


@lru_cache
def settings() -> Settings:
    return Settings()
