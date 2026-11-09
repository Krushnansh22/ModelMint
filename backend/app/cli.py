"""Small admin CLI for API keys.

Usage (from backend/):
    python -m app.cli create-key --name local-admin --scope admin
    python -m app.cli create-key --name iris --scope predict --model iris-classifier
    python -m app.cli list-keys
    python -m app.cli revoke-key <key-id>
"""
from __future__ import annotations

import argparse
import sys

from app.database import models as db_models
from app.database.database import SessionLocal, init_db
from app.security.auth import (
    create_api_key,
    hash_api_key,
    revoke_api_key,
)


def _resolve_model(db, id_or_slug: str) -> db_models.Model | None:
    return (
        db.query(db_models.Model)
        .filter((db_models.Model.id == id_or_slug) | (db_models.Model.slug == id_or_slug))
        .first()
    )


def cmd_create_key(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        model_id = None
        if args.model:
            model = _resolve_model(db, args.model)
            if model is None:
                print(f"error: model not found: {args.model}", file=sys.stderr)
                return 2
            model_id = model.id

        row, plaintext = create_api_key(
            db,
            name=args.name,
            scopes=args.scope,
            model_id=model_id,
            expires_in_days=args.expires_days,
        )
        print("API key created (shown once, store it now):")
        print(plaintext)
        print(f"id={row.id} prefix={row.prefix} scopes={','.join(row.scopes)}")
        return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        db.close()


def cmd_list_keys(_args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        rows = db.query(db_models.ApiKey).order_by(db_models.ApiKey.created_at.desc()).all()
        if not rows:
            print("no API keys")
            return 0
        for r in rows:
            state = "revoked" if r.revoked_at else "active"
            print(
                f"{r.id}  {r.prefix}…  {r.name!r}  scopes={','.join(r.scopes or [])}  {state}"
            )
        return 0
    finally:
        db.close()


def cmd_revoke_key(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        row = db.query(db_models.ApiKey).filter(db_models.ApiKey.id == args.key_id).first()
        if row is None:
            print(f"error: key not found: {args.key_id}", file=sys.stderr)
            return 2
        revoke_api_key(db, row)
        print(f"revoked {row.id}")
        return 0
    finally:
        db.close()


def cmd_hash_key(args: argparse.Namespace) -> int:
    print(hash_api_key(args.key))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="ModelMint admin CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create-key", help="create a scoped API key")
    p_create.add_argument("--name", required=True)
    p_create.add_argument(
        "--scope",
        action="append",
        default=[],
        help="repeatable: admin, models:read, models:write, predict, monitoring:read",
    )
    p_create.add_argument("--model", default=None, help="optional model id or slug to bind the key to")
    p_create.add_argument("--expires-days", type=int, default=None)
    p_create.set_defaults(func=cmd_create_key)

    p_list = sub.add_parser("list-keys", help="list API keys")
    p_list.set_defaults(func=cmd_list_keys)

    p_revoke = sub.add_parser("revoke-key", help="revoke an API key by id")
    p_revoke.add_argument("key_id")
    p_revoke.set_defaults(func=cmd_revoke_key)

    p_hash = sub.add_parser("hash-key", help="print the stored hash for a key (debugging)")
    p_hash.add_argument("key")
    p_hash.set_defaults(func=cmd_hash_key)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "create-key" and not args.scope:
        args.scope = ["predict"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
