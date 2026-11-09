"""Security + multi-format metadata: api_keys, audit_logs, request metadata.

Revision ID: 0002_security
Revises: 0001_initial
Create Date: 2026-11-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.database.models import GUID, JSONVariant

revision: str = "0002_security"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("prefix", sa.String(length=24), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("scopes", JSONVariant(), nullable=False),
        sa.Column("model_id", GUID(), sa.ForeignKey("models.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"], unique=True)
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_model_id", "api_keys", ["model_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("actor_key_id", GUID(), sa.ForeignKey("api_keys.id"), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("resource", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_audit_logs_actor_key_id", "audit_logs", ["actor_key_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])

    with op.batch_alter_table("models") as batch_op:
        batch_op.add_column(sa.Column("task_type", sa.String(), nullable=True, server_default="tabular"))
        batch_op.add_column(sa.Column("artifact_format", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("checksum", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("storage_backend", sa.String(), nullable=True, server_default="local"))
        batch_op.add_column(sa.Column("storage_key", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("framework_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("version", sa.BigInteger(), nullable=False, server_default="1"))
        batch_op.add_column(sa.Column("family_id", GUID(), nullable=True))
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime(), nullable=True))

    op.create_index("ix_models_checksum", "models", ["checksum"])
    op.create_index("ix_models_family_id", "models", ["family_id"])

    with op.batch_alter_table("inference_logs") as batch_op:
        batch_op.add_column(sa.Column("api_key_id", GUID(), nullable=True))
        batch_op.add_column(sa.Column("client_ip", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("request_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("task_type", sa.String(), nullable=True))

    op.create_index("ix_inference_logs_api_key_id", "inference_logs", ["api_key_id"])


def downgrade() -> None:
    op.drop_index("ix_inference_logs_api_key_id", table_name="inference_logs")
    with op.batch_alter_table("inference_logs") as batch_op:
        batch_op.drop_column("task_type")
        batch_op.drop_column("request_id")
        batch_op.drop_column("client_ip")
        batch_op.drop_column("api_key_id")

    op.drop_index("ix_models_family_id", table_name="models")
    op.drop_index("ix_models_checksum", table_name="models")
    with op.batch_alter_table("models") as batch_op:
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("family_id")
        batch_op.drop_column("version")
        batch_op.drop_column("framework_version")
        batch_op.drop_column("storage_key")
        batch_op.drop_column("storage_backend")
        batch_op.drop_column("checksum")
        batch_op.drop_column("artifact_format")
        batch_op.drop_column("task_type")

    op.drop_index("ix_audit_logs_timestamp", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_key_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_api_keys_model_id", table_name="api_keys")
    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
    op.drop_table("api_keys")
