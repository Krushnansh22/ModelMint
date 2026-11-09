"""Artifact storage abstraction.

Two backends:

- `LocalStorage` writes under `MODEL_STORAGE_DIR` (default `storage/`).
- `S3Storage` talks to any S3-compatible endpoint (AWS S3, MinIO) via boto3.

Serving needs a local path (sandboxed workers read the artifact from disk), so
`S3Storage.local_path()` downloads into a local cache and returns that path.
Model rows keep `storage_key` (object key) and `file_path` (local cache path).

boto3 is imported lazily so the default local/SQLite install needs no extra
dependency; it is listed in requirements.txt because Compose uses MinIO.
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings


class StorageBackend(ABC):
    name: str = "local"

    @abstractmethod
    def put_file(self, src_path: str, key: str) -> str:
        """Persist `src_path` under `key`; returns the local path for the file."""

    @abstractmethod
    def local_path(self, key: str) -> str:
        """Return a local filesystem path for `key`, fetching it if needed."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the object. Missing objects are ignored."""


class LocalStorage(StorageBackend):
    name = "local"

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are generated server-side (`<uuid>.<ext>`), but normalize anyway
        # so a malformed key can never escape the storage root.
        safe = os.path.basename(key)
        return self.root / safe

    def put_file(self, src_path: str, key: str) -> str:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(src_path).resolve() != dest:
            shutil.move(src_path, dest)
        return str(dest)

    def local_path(self, key: str) -> str:
        return str(self._path(key))

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass


class S3Storage(StorageBackend):
    name = "s3"

    def __init__(
        self,
        *,
        endpoint_url: str | None,
        bucket: str,
        access_key: str | None,
        secret_key: str | None,
        region: str,
        cache_dir: str,
    ) -> None:
        import boto3  # lazy: optional dependency
        from botocore.config import Config

        self.bucket = bucket
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / os.path.basename(key)

    def put_file(self, src_path: str, key: str) -> str:
        self._client.upload_file(src_path, self.bucket, key)
        cache = self._cache_path(key)
        if Path(src_path).resolve() != cache:
            shutil.move(src_path, cache)
        return str(cache)

    def local_path(self, key: str) -> str:
        cache = self._cache_path(key)
        if not cache.exists():
            self._client.download_file(self.bucket, key, str(cache))
        return str(cache)

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:  # noqa: BLE001 -- deletion is best-effort
            pass
        try:
            self._cache_path(key).unlink()
        except FileNotFoundError:
            pass


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is not None:
        return _backend

    settings = get_settings()
    if settings.storage_backend == "s3":
        _backend = S3Storage(
            endpoint_url=settings.s3_endpoint_url,
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
            cache_dir=os.path.join(settings.model_storage_dir, "_cache"),
        )
    else:
        _backend = LocalStorage(settings.model_storage_dir)
    return _backend


def reset_storage() -> None:
    """Test helper."""
    global _backend
    _backend = None


def ensure_local_path(row) -> str:
    """Return a local file for a model row, re-downloading S3 objects if the
    local cache was lost (e.g. after a container restart)."""
    if row.storage_backend == "s3" and row.storage_key:
        path = get_storage().local_path(row.storage_key)
        row.file_path = path
        return path
    return row.file_path
