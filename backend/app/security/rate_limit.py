"""Sliding-window rate limiting.

Default backend is in-process (fine for the single-worker demo). If
`REDIS_URL` is set and the `redis` package is importable, a Redis sorted-set
limiter is used instead so limits are shared across workers.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response, status

from app.config import get_settings
from app.security.auth import AuthContext, get_auth_context
from app.security.utils import client_ip


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float


class InMemoryLimiter:
    """Per-identity sliding window, guarded by a lock."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

    def check(self, identity: str, limit: int, window_seconds: int) -> LimitResult:
        now = time.monotonic()
        with self._lock:
            dq = self._events.setdefault(identity, deque())
            cutoff = now - window_seconds
            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= limit:
                retry_after = max(0.0, window_seconds - (now - dq[0]))
                return LimitResult(False, limit, 0, retry_after)

            dq.append(now)
            self._cleanup_locked(now, window_seconds)
            return LimitResult(True, limit, limit - len(dq), 0.0)

    def _cleanup_locked(self, now: float, window_seconds: int) -> None:
        # Drop idle identities at most once per window.
        if now - self._last_cleanup < window_seconds:
            return
        self._last_cleanup = now
        cutoff = now - window_seconds
        stale = [k for k, dq in self._events.items() if not dq or dq[-1] <= cutoff]
        for key in stale:
            self._events.pop(key, None)


_REDIS_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local oldest_score = tonumber(oldest[2])
  return {0, count, oldest_score}
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, window + 1)
return {1, count + 1, 0}
"""


class RedisLimiter:
    def __init__(self, url: str) -> None:
        import redis  # imported lazily so Redis stays optional

        self._client = redis.from_url(url, decode_responses=True)

    def check(self, identity: str, limit: int, window_seconds: int) -> LimitResult:
        now = time.time()
        key = f"modelmint:rl:{identity}"
        member = f"{now}:{uuid.uuid4().hex}"
        allowed, count, oldest = self._client.eval(
            _REDIS_SCRIPT, 1, key, now, window_seconds, limit, member
        )
        if int(allowed) == 1:
            return LimitResult(True, limit, max(0, limit - int(count)), 0.0)
        retry_after = max(0.0, window_seconds - (now - float(oldest)))
        return LimitResult(False, limit, 0, retry_after)


_limiter = None
_limiter_lock = threading.Lock()


def get_limiter():
    global _limiter
    if _limiter is not None:
        return _limiter
    with _limiter_lock:
        if _limiter is None:
            settings = get_settings()
            if settings.redis_url:
                try:
                    _limiter = RedisLimiter(settings.redis_url)
                except Exception:  # noqa: BLE001 -- fall back to local
                    _limiter = InMemoryLimiter()
            else:
                _limiter = InMemoryLimiter()
    return _limiter


def reset_limiter() -> None:
    """Test helper."""
    global _limiter
    with _limiter_lock:
        _limiter = None


def rate_limit(limit_attr: str, window_seconds: int):
    """Dependency factory. Keys by API-key id, falling back to client IP."""

    def dependency(
        request: Request,
        response: Response,
        ctx: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        limit = int(getattr(get_settings(), limit_attr))
        if limit <= 0:
            return ctx

        identity = f"key:{ctx.key.id}" if ctx.key else f"ip:{client_ip(request) or 'unknown'}"
        result = get_limiter().check(identity, limit, window_seconds)

        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Window"] = str(window_seconds)

        if not result.allowed:
            retry_after = max(1, int(result.retry_after + 0.999))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        return ctx

    return dependency


predict_rate_limit = rate_limit("rate_limit_predict_per_min", 60)
upload_rate_limit = rate_limit("rate_limit_upload_per_hour", 3600)
