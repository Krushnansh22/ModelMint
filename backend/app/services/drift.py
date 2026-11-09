"""Input drift detection (M5).

We do not store raw request bodies. Instead, every prediction writes a small
JSON-safe `input_summary` (per-feature mean/std/min/max for tabular models, or
text/image length statistics). The drift endpoint compares the older half of a
model's recent window against the newer half and reports a per-feature
z-score. A feature is flagged when |z| exceeds the threshold.

This is deliberately simple and dependency-free (numpy only): it catches
mean/variance shifts without retaining user data, and it is a natural first
step toward PSI/KS-based drift monitoring.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.database import models as db_models

DEFAULT_WINDOW = 100
DEFAULT_THRESHOLD = 3.0
MIN_SAMPLES = 10


def compute_input_summary(payload: dict[str, Any], task_type: Optional[str]) -> Optional[dict]:
    """Summarize one inference payload for later drift analysis."""
    task = (task_type or "tabular").lower()
    if task == "vision":
        image = payload.get("image_base64")
        if isinstance(image, str):
            return {"kind": "vision", "n": 1, "b64_length": len(image)}
        return None

    if task == "text":
        lengths = _text_lengths(payload)
        return _length_summary("text", lengths) if lengths else None

    if task == "embedding":
        if payload.get("text") is not None or payload.get("texts") is not None:
            lengths = _text_lengths(payload)
            return _length_summary("embedding_text", lengths) if lengths else None
        return _tabular_summary(_tabular_rows(payload))

    return _tabular_summary(_tabular_rows(payload))


def summarize_batch(instances: list[dict], task_type: Optional[str]) -> Optional[dict]:
    """Summarize a batch of payloads into one summary row."""
    task = (task_type or "tabular").lower()
    if task == "text":
        lengths: list[int] = []
        for instance in instances:
            lengths.extend(_text_lengths(instance))
        return _length_summary("text", lengths) if lengths else None

    if task == "embedding":
        lengths = []
        for instance in instances:
            lengths.extend(_text_lengths(instance))
        if lengths:
            return _length_summary("embedding_text", lengths)
        rows: list[list[float]] = []
        for instance in instances:
            rows.extend(_tabular_rows(instance))
        return _tabular_summary(rows)

    if task == "vision":
        sizes = [
            len(instance["image_base64"])
            for instance in instances
            if isinstance(instance.get("image_base64"), str)
        ]
        if not sizes:
            return None
        return {
            "kind": "vision",
            "n": len(sizes),
            "b64_length": float(np.mean(sizes)),
        }

    rows = []
    for instance in instances:
        rows.extend(_tabular_rows(instance))
    return _tabular_summary(rows)


def compute_drift(
    db: Session,
    model_row: db_models.Model,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    window = max(MIN_SAMPLES, min(int(window), 1000))
    logs = (
        db.query(db_models.InferenceLog)
        .filter(
            db_models.InferenceLog.model_id == model_row.id,
            db_models.InferenceLog.input_summary.isnot(None),
        )
        .order_by(db_models.InferenceLog.timestamp.desc())
        .limit(window)
        .all()
    )
    summaries = [log.input_summary for log in logs if isinstance(log.input_summary, dict)]

    result = {
        "model_id": model_row.id,
        "version": model_row.version,
        "task_type": model_row.task_type or "tabular",
        "status": "INSUFFICIENT_DATA",
        "threshold": threshold,
        "window": window,
        "baseline_count": 0,
        "current_count": 0,
        "features": [],
    }
    if len(summaries) < MIN_SAMPLES:
        result["reason"] = (
            f"need at least {MIN_SAMPLES} summarized requests, have {len(summaries)}"
        )
        return result

    half = len(summaries) // 2
    # `summaries` is newest-first; the current window is the newest half and
    # the baseline is the older half.
    current = summaries[:half]
    baseline = summaries[half:]
    result["baseline_count"] = len(baseline)
    result["current_count"] = len(current)

    kind = current[0].get("kind")
    if kind == "tabular":
        features = _tabular_drift(model_row, baseline, current, threshold)
    elif kind in ("text", "embedding_text"):
        features = _length_drift(baseline, current, threshold, kind)
    elif kind == "vision":
        features = _scalar_drift(
            "b64_length",
            [float(s.get("b64_length", 0.0)) for s in baseline],
            [float(s.get("b64_length", 0.0)) for s in current],
            threshold,
        )
    else:
        features = []

    result["features"] = features
    result["status"] = "DRIFT" if any(f["drift"] for f in features) else "OK"
    return result


def _tabular_drift(model_row, baseline, current, threshold) -> list[dict]:
    baseline_means = [s.get("mean") for s in baseline if isinstance(s.get("mean"), list)]
    current_means = [s.get("mean") for s in current if isinstance(s.get("mean"), list)]
    if not baseline_means or not current_means:
        return []
    width = min(len(row) for row in baseline_means + current_means)
    if width == 0:
        return []

    baseline_matrix = np.asarray([row[:width] for row in baseline_means], dtype=float)
    current_matrix = np.asarray([row[:width] for row in current_means], dtype=float)

    baseline_std_rows = [
        s.get("std") for s in baseline if isinstance(s.get("std"), list)
    ]
    fallback_std = (
        np.asarray([row[:width] for row in baseline_std_rows], dtype=float).mean(axis=0)
        if baseline_std_rows
        else np.zeros(width)
    )

    base_mean = baseline_matrix.mean(axis=0)
    base_std = baseline_matrix.std(axis=0)
    # A constant baseline feature has zero between-row variance; fall back to
    # the mean of the within-request standard deviations when available.
    base_std = np.where(base_std > 1e-9, base_std, fallback_std)
    base_std = np.where(base_std > 1e-9, base_std, 1e-6)
    cur_mean = current_matrix.mean(axis=0)
    z_scores = np.abs(cur_mean - base_mean) / base_std

    names = _feature_names(model_row, width)
    return [
        {
            "feature": names[i],
            "baseline_mean": round(float(base_mean[i]), 6),
            "current_mean": round(float(cur_mean[i]), 6),
            "z_score": round(float(z_scores[i]), 4),
            "drift": bool(z_scores[i] > threshold),
        }
        for i in range(width)
    ]


def _length_drift(baseline, current, threshold, kind) -> list[dict]:
    base = [float(s.get("length_mean", 0.0)) for s in baseline]
    cur = [float(s.get("length_mean", 0.0)) for s in current]
    return _scalar_drift(f"{kind}_length", base, cur, threshold)


def _scalar_drift(name, baseline, current, threshold) -> list[dict]:
    base = np.asarray(baseline, dtype=float)
    cur = np.asarray(current, dtype=float)
    if base.size == 0 or cur.size == 0:
        return []
    base_mean = float(base.mean())
    base_std = float(base.std())
    if base_std <= 1e-9:
        base_std = 1e-6
    cur_mean = float(cur.mean())
    z = abs(cur_mean - base_mean) / base_std
    return [
        {
            "feature": name,
            "baseline_mean": round(base_mean, 6),
            "current_mean": round(cur_mean, 6),
            "z_score": round(float(z), 4),
            "drift": bool(z > threshold),
        }
    ]


def _feature_names(model_row, width: int) -> list[str]:
    schema = model_row.input_schema or {}
    fields = schema.get("fields") or []
    names = [f.get("name") for f in fields if isinstance(f, dict) and f.get("name")]
    if len(names) >= width:
        return names[:width]
    return [f"feature_{i}" for i in range(width)]


def _tabular_rows(payload: dict[str, Any]) -> list[list[float]]:
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return []
    if isinstance(features[0], list):
        rows = features
    else:
        rows = [features]
    cleaned = []
    for row in rows:
        if not isinstance(row, list):
            continue
        try:
            cleaned.append([float(v) for v in row])
        except (TypeError, ValueError):
            continue
    return cleaned


def _text_lengths(payload: dict[str, Any]) -> list[int]:
    lengths: list[int] = []
    text = payload.get("text")
    prompt = payload.get("prompt")
    texts = payload.get("texts")
    if isinstance(text, str):
        lengths.append(len(text))
    if isinstance(prompt, str):
        lengths.append(len(prompt))
    if isinstance(texts, list):
        lengths.extend(len(t) for t in texts if isinstance(t, str))
    return lengths


def _tabular_summary(rows: list[list[float]]) -> Optional[dict]:
    if not rows:
        return None
    width = min(len(row) for row in rows)
    if width == 0:
        return None
    matrix = np.asarray([row[:width] for row in rows], dtype=float)
    return {
        "kind": "tabular",
        "n": int(matrix.shape[0]),
        "mean": [round(float(v), 6) for v in matrix.mean(axis=0)],
        "std": [round(float(v), 6) for v in matrix.std(axis=0)],
        "min": [round(float(v), 6) for v in matrix.min(axis=0)],
        "max": [round(float(v), 6) for v in matrix.max(axis=0)],
    }


def _length_summary(kind: str, lengths: list[int]) -> dict:
    array = np.asarray(lengths, dtype=float)
    return {
        "kind": kind,
        "n": int(array.size),
        "length_mean": round(float(array.mean()), 6),
        "length_std": round(float(array.std()), 6),
        "length_min": int(array.min()),
        "length_max": int(array.max()),
    }
