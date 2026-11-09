"""Baseline schema (pre-security): models + inference_logs.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.database.models import GUID, JSONVariant

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("framework", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("input_schema", JSONVariant(), nullable=True),
        sa.Column("output_schema", JSONVariant(), nullable=True),
        sa.Column("schema_confidence", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("load_time_ms", sa.Float(), nullable=True),
        sa.Column("memory_usage_mb", sa.Float(), nullable=True),
        sa.Column("metadata_json", JSONVariant(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_models_slug", "models", ["slug"], unique=True)

    op.create_table(
        "inference_logs",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("model_id", GUID(), sa.ForeignKey("models.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("ix_inference_logs_model_id", "inference_logs", ["model_id"])
    op.create_index("ix_inference_logs_timestamp", "inference_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_inference_logs_timestamp", table_name="inference_logs")
    op.drop_index("ix_inference_logs_model_id", table_name="inference_logs")
    op.drop_table("inference_logs")
    op.drop_index("ix_models_slug", table_name="models")
    op.drop_table("models")
