"""API-key generation, hashing, lookup, and FastAPI auth dependencies.

Keys are high-entropy random tokens. We store `HMAC-SHA256(pepper, key)`
rather than the plaintext, and show the plaintext exactly once at
creation/rotation time. `prefix` is stored only so the UI can display a
recognizable fragment.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import models as db_models
from app.database.database import get_db
from app.security.audit import record_audit
from app.security.utils import client_ip, get_request_id

API_KEY_HEADER = "X-API-Key"
KEY_PREFIX = "mm_live_"
PREFIX_LENGTH = 16  # "mm_live_" + 8 chars

VALID_SCOPES = {
    "admin",
    "models:read",
    "models:write",
    "predict",
    "monitoring:read",
}

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    """Resolved authentication state for one request."""

    key: Optional[db_models.ApiKey]
    scopes: frozenset[str]
    is_admin: bool
    model_id: Optional[str]
    raw_key_present: bool

    @property
    def key_id(self) -> Optional[str]:
        return self.key.id if self.key else None


ANONYMOUS = AuthContext(key=None, scopes=frozenset(), is_admin=False, model_id=None, raw_key_present=False)


def hash_api_key(raw_key: str) -> str:
    settings = get_settings()
    return hmac.new(settings.api_key_pepper.encode("utf-8"), raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (plaintext, prefix, hash)."""
    raw = f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"
    return raw, raw[:PREFIX_LENGTH], hash_api_key(raw)


def create_api_key(
    db: Session,
    *,
    name: str,
    scopes: Iterable[str],
    model_id: Optional[str] = None,
    expires_in_days: Optional[int] = None,
) -> tuple[db_models.ApiKey, str]:
    clean_scopes = sorted({s for s in scopes if s})
    unknown = set(clean_scopes) - VALID_SCOPES
    if unknown:
        raise ValueError(f"unknown scopes: {', '.join(sorted(unknown))}")
    if not clean_scopes:
        raise ValueError("at least one scope is required")

    raw, prefix, digest = generate_api_key()
    expires_at = None
    if expires_in_days is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

    row = db_models.ApiKey(
        name=name,
        prefix=prefix,
        key_hash=digest,
        scopes=clean_scopes,
        model_id=model_id,
        expires_at=expires_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, raw


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_active(row: db_models.ApiKey) -> bool:
    if row.revoked_at is not None:
        return False
    expires_at = _as_utc(row.expires_at)
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        return False
    return True


def _extract_raw_key(
    api_key: Optional[str],
    bearer: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    if api_key:
        return api_key.strip()
    if bearer and bearer.credentials:
        return bearer.credentials.strip()
    return None


def _unauthorized(detail: str = "Missing or invalid API key") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"message": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _enforce(ctx: AuthContext, required: Iterable[str]) -> AuthContext:
    required_set = set(required)
    if not required_set:
        return ctx
    if ctx.key is None:
        raise _unauthorized()
    if ctx.is_admin:
        return ctx
    missing = required_set - set(ctx.scopes)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Insufficient scope for this operation",
                "required": sorted(required_set),
                "granted": sorted(ctx.scopes),
            },
        )
    return ctx


def get_auth_context(
    request: Request,
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    db: Session = Depends(get_db),
) -> AuthContext:
    raw = _extract_raw_key(api_key, bearer)
    if not raw:
        return ANONYMOUS

    digest = hash_api_key(raw)
    row = db.query(db_models.ApiKey).filter(db_models.ApiKey.key_hash == digest).first()
    if row is None or not _is_active(row):
        record_audit(
            db,
            action="auth.failure",
            status="FAILURE",
            request=request,
            detail="invalid, revoked, or expired API key",
        )
        raise _unauthorized()

    # Throttle last_used_at writes to at most once per minute per key.
    now = datetime.now(timezone.utc)
    last_used = _as_utc(row.last_used_at)
    if last_used is None or (now - last_used).total_seconds() > 60:
        row.last_used_at = now
        try:
            db.commit()
        except Exception:  # noqa: BLE001 -- never fail a request over telemetry
            db.rollback()

    scopes = frozenset(row.scopes or [])
    return AuthContext(
        key=row,
        scopes=scopes,
        is_admin="admin" in scopes,
        model_id=row.model_id,
        raw_key_present=True,
    )


def require_scopes(*required: str):
    """Dependency factory: require all of `required` (admin bypasses)."""

    def dependency(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        return _enforce(ctx, required)

    return dependency


def require_scopes_if_enabled(flag_name: str, *required: str):
    """Dependency factory used for read/predict routes whose auth is
    configurable (REQUIRE_AUTH_FOR_READS / REQUIRE_AUTH_FOR_PREDICT)."""

    def dependency(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if not getattr(get_settings(), flag_name):
            return ctx
        return _enforce(ctx, required)

    return dependency


def ensure_model_binding(ctx: AuthContext, model_id: str) -> None:
    """A key bound to one model may not call another model's predict route."""
    if ctx.key is not None and ctx.model_id and ctx.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "This API key is bound to a different model"},
        )


def bootstrap_admin_key(db: Session) -> Optional[str]:
    """Create the env-provided bootstrap admin key if it doesn't exist.

    Returns the plaintext key when it was created (for a startup log line),
    otherwise None. The value already exists in the environment, so logging
    it is no worse than the config itself.
    """
    settings = get_settings()
    raw = settings.bootstrap_admin_key
    if not raw:
        return None
    digest = hash_api_key(raw)
    existing = db.query(db_models.ApiKey).filter(db_models.ApiKey.key_hash == digest).first()
    if existing is not None:
        return None
    row = db_models.ApiKey(
        name="bootstrap-admin",
        prefix=raw[:PREFIX_LENGTH],
        key_hash=digest,
        scopes=["admin"],
    )
    db.add(row)
    db.commit()
    return raw


def revoke_api_key(db: Session, row: db_models.ApiKey) -> None:
    row.revoked_at = datetime.now(timezone.utc)
    db.commit()


def rotate_api_key(db: Session, row: db_models.ApiKey) -> tuple[db_models.ApiKey, str]:
    """Revoke the old key and issue a new one with the same name/scopes/binding."""
    revoke_api_key(db, row)
    return create_api_key(
        db,
        name=row.name,
        scopes=row.scopes or [],
        model_id=row.model_id,
        expires_in_days=None,
    )
