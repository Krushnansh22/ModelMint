"""
Centralized, 12-factor configuration (pydantic-settings).

Every environment variable the app reads should be declared here. This keeps
routes/services free of scattered `os.environ.get(...)` calls and makes the
Docker/Compose story straightforward: one `.env` file, one Settings object.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # --- app ---------------------------------------------------------------
    app_name: str = "ModelMint"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- database ----------------------------------------------------------
    database_url: str = "sqlite:///./storage/modelmint.db"

    # --- storage -----------------------------------------------------------
    # `local` writes under MODEL_STORAGE_DIR; `s3` is wired in M4.
    storage_backend: Literal["local", "s3"] = "local"
    model_storage_dir: str = "storage"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "modelmint"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"

    # --- security ----------------------------------------------------------
    # Server-side pepper mixed into the HMAC used to store API keys. Set a
    # long random value in production; the default is dev-only.
    api_key_pepper: str = "dev-only-change-me"
    # If set, an admin key with this value is created on first startup (only
    # when no admin key already exists). Convenient for local demos.
    bootstrap_admin_key: str | None = None
    require_auth_for_predict: bool = True
    require_auth_for_reads: bool = False

    # --- rate limits -------------------------------------------------------
    rate_limit_predict_per_min: int = 120
    rate_limit_upload_per_hour: int = 20
    # Optional: share the limiter across workers. Falls back to in-memory.
    redis_url: str | None = None

    # --- uploads -----------------------------------------------------------
    max_upload_mb: int = 500

    # --- model manager / sandbox ------------------------------------------
    max_resident_models: int = 5
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit_mb: int = 2048
    sandbox_cpu_limit_seconds: int = 30
    # Per-request timeout for the persistent sandboxed worker (M2).
    inference_timeout_seconds: int = 30

    # --- CORS --------------------------------------------------------------
    # Comma-separated list, e.g. "http://localhost:5173,https://app.example.com"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
