"""
Pydantic types shared across the adapter interface, services, and API layer.
These are plain data-transfer objects -- NOT the SQLAlchemy ORM models
(see app/database/models.py for those).
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SchemaConfidence(str, Enum):
    STRICT = "STRICT"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class TaskType(str, Enum):
    TABULAR = "tabular"
    TEXT = "text"
    VISION = "vision"
    EMBEDDING = "embedding"


class InputContract(str, Enum):
    """What shape of JSON body a model expects."""
    FEATURES = "features"          # {"features": [...]}
    TEXT = "text"                  # {"text": "..."} / {"texts": [...]}
    IMAGE = "image"                # {"image_base64": "..."}
    INSTANCES = "instances"        # {"instances": [ {...}, ... ]}


class ValidationStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class ModelStatus(str, Enum):
    UPLOADING = "UPLOADING"
    ANALYZING = "ANALYZING"
    READY = "READY"
    READY_UNVALIDATED = "READY_UNVALIDATED"
    FAILED = "FAILED"


class FieldDescriptor(BaseModel):
    name: Optional[str] = None
    dtype: Optional[str] = None   # e.g. 'float32', 'int64'
    shape: Optional[list[Optional[int]]] = None  # None entries = dynamic dim


class SchemaDescriptor(BaseModel):
    """Describes model input/output shape. `fields` is empty for INFERRED
    schemas that only know a feature count (see `feature_count`)."""
    fields: list[FieldDescriptor] = Field(default_factory=list)
    feature_count: Optional[int] = None


class ValidationResult(BaseModel):
    status: ValidationStatus
    schema_confidence: SchemaConfidence
    reason: str


class ModelMetadata(BaseModel):
    """Only framework-applicable fields should be populated -- omit
    (leave None / default) fields that don't apply rather than null-filling."""
    model_config = {"protected_namespaces": ()}

    framework: str
    model_type: Optional[str] = None
    task_type: Optional[str] = None
    input_contract: Optional[str] = None
    artifact_format: Optional[str] = None
    framework_version: Optional[str] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    estimator_params: Optional[dict[str, Any]] = None   # sklearn only
    onnx_ir_version: Optional[int] = None                # onnx only
    onnx_producer: Optional[str] = None                  # onnx only
    extra: dict[str, Any] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    predictions: Any
    raw_output_shape: Optional[list[int]] = None
