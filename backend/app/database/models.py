"""SQLAlchemy ORM models -- mirrors the data model in the build guide (section 7).

Security additions (M1): ApiKey + AuditLog, plus key/request metadata on
InferenceLog. Multi-format additions (M3) are present as nullable columns so
the schema is forward-compatible.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, CHAR, Column, DateTime, Float, ForeignKey, String, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.types import JSON

from app.database.database import Base


def _uuid_str():
    return str(uuid.uuid4())


def _utcnow():
    return datetime.now(timezone.utc)


class GUID(TypeDecorator):
    """Platform-independent UUID column: native UUID on Postgres, CHAR(36) elsewhere."""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID())
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return str(value)


class JSONVariant(TypeDecorator):
    """JSONB on Postgres, plain JSON elsewhere (SQLite)."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class Model(Base):
    __tablename__ = "models"

    id = Column(GUID(), primary_key=True, default=_uuid_str)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    framework = Column(String, nullable=False)          # 'sklearn' | 'onnx' | ...
    model_type = Column(String, nullable=True)           # e.g. 'RandomForestClassifier'
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)

    input_schema = Column(JSONVariant, nullable=True)
    output_schema = Column(JSONVariant, nullable=True)
    schema_confidence = Column(String, nullable=True)    # STRICT | INFERRED | UNKNOWN

    status = Column(String, nullable=False, default="UPLOADING")
    failure_reason = Column(Text, nullable=True)

    load_time_ms = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)

    metadata_json = Column(JSONVariant, nullable=True)   # ModelMetadata dump (framework-specific fields)

    # --- multi-format / artifact metadata (M3) ----------------------------
    task_type = Column(String, nullable=True, default="tabular")
    artifact_format = Column(String, nullable=True)
    checksum = Column(String(64), nullable=True, index=True)
    storage_backend = Column(String, nullable=True, default="local")
    storage_key = Column(String, nullable=True)
    framework_version = Column(String, nullable=True)
    version = Column(BigInteger, nullable=False, default=1)
    family_id = Column(GUID(), nullable=True, index=True)
    # Versioning / traffic (M5): one family can have several versions. The
    # active versions with traffic_weight > 0 receive traffic through the
    # /api/families/{family_id}/predict alias.
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    traffic_weight = Column(Float, nullable=False, default=1.0)
    promoted_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(GUID(), primary_key=True, default=_uuid_str)
    model_id = Column(GUID(), ForeignKey("models.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)
    status = Column(String, nullable=False)   # SUCCESS | FAILED
    latency_ms = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)

    # --- request metadata (M1) --------------------------------------------
    api_key_id = Column(GUID(), ForeignKey("api_keys.id"), nullable=True, index=True)
    client_ip = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    # M5: denormalized version + a small JSON-safe input summary used by
    # drift detection (per-feature mean/std/min/max, or text length).
    model_version = Column(BigInteger, nullable=True)
    input_summary = Column(JSONVariant, nullable=True)


class ApiKey(Base):
    """A scoped API key. Only the HMAC hash is stored; plaintext is shown once."""
    __tablename__ = "api_keys"

    id = Column(GUID(), primary_key=True, default=_uuid_str)
    name = Column(String, nullable=False)
    prefix = Column(String(24), nullable=False, unique=True, index=True)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    scopes = Column(JSONVariant, nullable=False, default=list)
    model_id = Column(GUID(), ForeignKey("models.id"), nullable=True, index=True)

    created_at = Column(DateTime, default=_utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    request_count = Column(BigInteger, nullable=False, default=0)


class AuditLog(Base):
    """Security-relevant events: key lifecycle, uploads, deletes, auth failures."""
    __tablename__ = "audit_logs"

    id = Column(GUID(), primary_key=True, default=_uuid_str)
    actor_key_id = Column(GUID(), ForeignKey("api_keys.id"), nullable=True, index=True)
    action = Column(String, nullable=False, index=True)
    resource = Column(String, nullable=True)
    status = Column(String, nullable=False)   # SUCCESS | FAILURE | DENIED
    ip_address = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    detail = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)
