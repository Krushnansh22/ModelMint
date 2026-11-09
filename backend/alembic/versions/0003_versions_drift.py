"""Model versioning, A/B traffic weights, and drift summaries (M5).

Revision ID: 0003_versions_drift
Revises: 0002_security
Create Date: 2026-11-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.database.models import GUID, JSONVariant

revision: str = "0003_versions_drift"
down_revision: Union[str, None] = "0002_security"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("models") as batch_op:
        batch_op.add_column(
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true"))
        )
        batch_op.add_column(
            sa.Column("traffic_weight", sa.Float(), nullable=False, server_default="1.0")
        )
        batch_op.add_column(sa.Column("promoted_at", sa.DateTime(), nullable=True))

    op.create_index("ix_models_is_active", "models", ["is_active"])

    with op.batch_alter_table("inference_logs") as batch_op:
        batch_op.add_column(sa.Column("model_version", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("input_summary", JSONVariant(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("inference_logs") as batch_op:
        batch_op.drop_column("input_summary")
        batch_op.drop_column("model_version")

    op.drop_index("ix_models_is_active", table_name="models")
    with op.batch_alter_table("models") as batch_op:
        batch_op.drop_column("promoted_at")
        batch_op.drop_column("traffic_weight")
        batch_op.drop_column("is_active")
