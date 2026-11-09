"""Request helpers shared by auth/audit/rate limiting."""
from __future__ import annotations

from typing import Optional

from fastapi import Request


def client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def get_request_id(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    return getattr(request.state, "request_id", None)
