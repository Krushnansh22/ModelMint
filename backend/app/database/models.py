"""SQLAlchemy ORM models -- mirrors the data model in the build guide (section 7)."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, BigInteger, Text, DateTime, ForeignKey
from sqlalchemy.types import JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy import TypeDecorator, CHAR

from app.database.database import Base


def _uuid_str():
    return str(uuid.uuid4())


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
    framework = Column(String, nullable=False)          # 'sklearn' | 'onnx'
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

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(GUID(), primary_key=True, default=_uuid_str)
    model_id = Column(GUID(), ForeignKey("models.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    status = Column(String, nullable=False)   # SUCCESS | FAILED
    latency_ms = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
