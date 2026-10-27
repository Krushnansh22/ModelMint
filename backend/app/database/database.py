"""
Database engine/session setup.

Defaults to a local SQLite file so the project runs out-of-the-box with zero
external services. Set DATABASE_URL to a Postgres DSN
(e.g. postgresql+psycopg2://user:pass@localhost:5432/modelmint) for the
production target described in the build guide -- the ORM layer is
Postgres-compatible (JSONB columns fall back to plain JSON on SQLite
automatically via a small TypeDecorator, see database/models.py).
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./storage/modelmint.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.database import models  # noqa: F401  (register tables on Base.metadata)
    Base.metadata.create_all(bind=engine)
