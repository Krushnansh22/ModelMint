"""
Database engine/session setup + Alembic bootstrap.

Defaults to a local SQLite file so the project runs out-of-the-box with zero
external services. Set DATABASE_URL to a Postgres DSN for the Compose/prod
target; the ORM layer is Postgres-compatible (JSONB falls back to JSON on
SQLite via a TypeDecorator, see database/models.py).

`init_db()` runs Alembic rather than `create_all()` so schema changes (new
security tables/columns) apply to existing databases. If the DB predates
Alembic (tables exist but no `alembic_version`), it is stamped at the
baseline revision first, then upgraded.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()
DATABASE_URL = settings.database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

BACKEND_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"
ALEMBIC_DIR = BACKEND_DIR / "alembic"
BASELINE_REVISION = "0001_initial"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create/upgrade the schema. Safe to call on every startup."""
    _ensure_sqlite_parent_dir()
    _run_migrations()


def _ensure_sqlite_parent_dir() -> None:
    if DATABASE_URL.startswith("sqlite:///"):
        path = DATABASE_URL.removeprefix("sqlite:///")
        if path and path != ":memory:":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

    tables = inspect(engine).get_table_names()
    if "models" in tables and "alembic_version" not in tables:
        # Pre-Alembic database: treat the existing schema as the baseline,
        # then let upgrade() apply the security migration on top.
        command.stamp(cfg, BASELINE_REVISION)

    command.upgrade(cfg, "head")
