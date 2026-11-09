# ModelMint

Upload a trained ML model (sklearn `.pkl`/`.joblib`, ONNX `.onnx`, XGBoost,
LightGBM, CatBoost, TorchScript, Keras, TFLite, a zipped Hugging Face bundle,
or GGUF) → it's auto-detected, sandbox-validated, and served behind a generic
inference API with live monitoring. Built per the accompanying engineering
build guide: sandboxed pickle loading, LRU model eviction, three-tier schema
confidence, and a single generic `/predict` route (Option A from the guide).

The current build adds **scoped API-key security** (hashed keys, per-key
rate limits, audit logs, streamed uploads with checksums), **multi-format
model serving** (task-aware tabular/text/vision/embedding contracts, batch
predict), and **model governance** (versions, promote/rollback, A/B traffic
splits, and input drift detection). See `PROGRESS.md` for the live build
tracker.

## Supported formats

| Framework | Extensions | Notes |
|---|---|---|
| sklearn | `.pkl`, `.joblib` | Sandboxed |
| ONNX | `.onnx` | ONNX Runtime |
| XGBoost | `.json`, `.ubj`, `.xgb` | Needs `requirements-formats.txt` |
| LightGBM | `.txt`, `.lgbm`, `.model` | Needs `requirements-formats.txt` |
| CatBoost | `.cbm` | Needs `requirements-formats.txt` |
| TorchScript | `.pt`, `.pth`, `.torchscript` | Sandboxed; needs `torch` |
| Keras | `.keras`, `.h5` | Sandboxed; needs `requirements-dl.txt` |
| TFLite | `.tflite` | Sandboxed; `tensorflow` or `tflite-runtime` |
| Hugging Face | `.zip` bundle | Sandboxed; `config.json` + weights; never `trust_remote_code` |
| GGUF | `.gguf` | Sandboxed; stretch format; needs `requirements-gguf.txt` |

`GET /api/formats` reports the live list and whether each optional package is
installed. Raw `state_dict`/`.safetensors` files are intentionally not
servable without an architecture; export to TorchScript or upload a zipped HF
bundle instead. A `.zip` upload is only accepted when it contains
`config.json` plus a recognized weights file.

Every phase in the build guide's Definition of Done (§11) is implemented and
was exercised end-to-end against real fixture models during development —
see "Verifying it works" below.

## Stack

- **Backend**: FastAPI + SQLAlchemy (SQLite by default, Postgres-ready), scikit-learn, ONNX Runtime
- **Frontend**: React + Vite, no UI framework, plain CSS design tokens
- **Storage**: local filesystem (`backend/storage/`) or S3/MinIO (`STORAGE_BACKEND=s3`)

## Quickstart

### Backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
pip install -r requirements-formats.txt --break-system-packages  # trees + TorchScript
pip install -r requirements-dl.txt --break-system-packages       # Keras/TFLite/HF (large)
pip install -r requirements-gguf.txt --break-system-packages     # GGUF (stretch)
cp .env.example .env                                       # then set API_KEY_PEPPER
python -m uvicorn app.main:app --reload --port 8000
```

Runs against a local SQLite file (`backend/storage/modelmint.db`) with zero
setup. Schema changes are applied automatically via Alembic on startup; an
existing pre-Alembic DB is stamped at the baseline and upgraded.

Create an admin key (or set `BOOTSTRAP_ADMIN_KEY` in `.env` before startup):

```bash
python -m app.cli create-key --name local-admin --scope admin
```

Runs against a local SQLite file with zero setup. To point at Postgres
instead:

```bash
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/modelmint"
```

API docs: http://localhost:8000/docs (generic route shape only — see
"Swagger" note below).

### Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api to :8000
```

### Generate test fixture models (optional, for trying uploads)

```bash
cd backend
python tests/test_adapters/make_fixtures.py
# writes tests/test_adapters/fixtures/iris_rf.pkl and iris_rf.onnx
```

Create a key for uploads and inference:

```bash
cd backend
# copy the printed mm_live_... value
python -m app.cli create-key --name admin --scope admin
export ADMIN_KEY=mm_live_...

python -m app.cli create-key --name demo --scope predict --scope models:write
# copy the printed mm_live_... value into DEMO_KEY below
```

Upload either file through the UI or:

```bash
curl -X POST http://localhost:8000/api/models \
  -H "X-API-Key: $DEMO_KEY" \
  -F "file=@tests/test_adapters/fixtures/iris_rf.pkl" -F "name=iris_classifier"

curl -X POST http://localhost:8000/api/models/iris-classifier/predict \
  -H "X-API-Key: $DEMO_KEY" \
  -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
```

### Security model (M1)

- **Scoped API keys** — `admin`, `models:read`, `models:write`, `predict`,
  `monitoring:read`. Send via `X-API-Key: mm_live_...` or
  `Authorization: Bearer mm_live_...`.
- **Hash-only storage** — keys are stored as `HMAC-SHA256(API_KEY_PEPPER, key)`;
  plaintext is shown once at creation/rotation.
- **Optional model binding** — a key can be restricted to a single model.
- **Rate limits** — per-key sliding window (per-IP fallback); in-memory by
  default, Redis when `REDIS_URL` is set. Returns `429` + `Retry-After`.
- **Audit log** — key create/rotate/revoke, uploads, deletes, auth failures.
- **Uploads** — streamed to disk, capped by `MAX_UPLOAD_MB`, SHA-256 checksum,
  duplicate detection.
- **Reads** are public by default; set `REQUIRE_AUTH_FOR_READS=true` to
  require `models:read` / `monitoring:read`.

The dashboard's **API keys** page lets you paste an admin key, create scoped
keys, rotate/revoke them, and view the audit feed.

### Model versions, A/B traffic, and drift (M5)

Every upload is a version in a model family. The first upload is `v1` and is
active; later uploads are inactive until promoted. The Model Details page
shows the family, lets you upload a new version, promote or roll back, and
edit traffic weights. The same operations are available over HTTP:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models/{id_or_slug}/versions` | list family versions |
| `POST` | `/api/models/{id_or_slug}/versions` | upload a new version (`file`, optional `name`) |
| `POST` | `/api/models/{id_or_slug}/promote` | make this version active |
| `POST` | `/api/models/{id_or_slug}/rollback` | activate the previous servable version |
| `POST` | `/api/models/{id_or_slug}/traffic` | set `{"weights": {"<version id/slug>": 0.7, ...}}` |
| `GET` | `/api/models/{id_or_slug}/drift` | per-feature drift z-scores |
| `POST` | `/api/families/{family_id}/predict` | serve the family alias using traffic weights |

Traffic weights are normalized to 1.0. The family alias is the endpoint to
use for A/B tests; `/api/models/{id}/predict` always serves that exact
version. A version upload can also be made directly with
`POST /api/models` and the `family` form field.

Drift detection stores no raw request bodies. Each prediction records a small
JSON summary (per-feature mean/std/min/max for tabular models, text/image
length statistics otherwise), then compares the older half of the recent
window against the newer half. A feature is flagged when `|z| > threshold`
(default `3.0`); fewer than 10 summarized requests returns
`INSUFFICIENT_DATA`.

### Docker Compose (Postgres + MinIO + Redis)

```bash
cp .env.docker.example .env      # set API_KEY_PEPPER (required)
docker compose up --build
# frontend: http://localhost:8080
# backend:  http://localhost:8000
# MinIO console: http://localhost:9001 (minioadmin / minioadmin)
```

The backend stores artifacts in MinIO (`S3Storage`) and downloads them into a
local cache for sandboxed serving. Use `BACKEND_TARGET=full` for the
XGBoost/LightGBM/CatBoost/TorchScript packages, `dl` for TensorFlow/Keras/
TFLite/Hugging Face, or `gguf` for llama-cpp-python:

```bash
BACKEND_TARGET=full docker compose up --build
BACKEND_TARGET=dl   docker compose up --build
BACKEND_TARGET=gguf docker compose up --build
```

`GET /metrics` exposes Prometheus text metrics; `/health` and `/ready` are
liveness/readiness probes. Set `LOG_JSON=true` (Compose does) for structured
JSON logs.

## Observability

| Endpoint | Purpose |
|---|---|
| `GET /health` | process liveness |
| `GET /ready` | database connectivity |
| `GET /metrics` | Prometheus text (models, inference counts, latency sum, active keys) |

Every response carries `X-Request-ID` (or echoes an incoming one), and
rate-limited responses include `X-RateLimit-*` + `Retry-After`.

### Run tests

```bash
cd backend
python tests/test_adapters/make_fixtures.py   # fixtures are gitignored, generate once
python -m pytest tests/ -q
```

Covers all adapters (detect/load/validate/predict), sandbox isolation
(a legitimate model loads through the sandbox, a pickle that tries to open a
network socket during unpickling is blocked, and a pickle that hangs forever
is killed on timeout), API-key security, upload hardening, the M3
registry/task-contract/batch tests, and the M5 versioning/promote/rollback/
traffic/drift tests.

## What's implemented (build guide checklist, §11)

- [x] Upload sklearn, ONNX, XGBoost, LightGBM, CatBoost, TorchScript, Keras,
      TFLite, Hugging Face bundles, and GGUF artifacts through the frontend
      (optional packages via the requirements packs)
- [x] Framework/type auto-detected for all formats via declarative
      `AdapterSpec` entries (`adapters/registry.py`, the only place framework
      branching is allowed)
- [x] Pickle load **and serving** happen in sandboxed subprocesses with
      network egress blocked. Upload-time introspection returns JSON-safe
      schema/metadata; serving uses a persistent worker per resident model
      (`app/services/sandboxed_adapter.py`), so untrusted pickle code never
      runs in the API process (verified by `test_sandbox_isolation.py`).
      Keras/TFLite/HF/GGUF use the same sandboxed serving path
- [x] Validation runs synthetic-input test inference, assigns a schema
      confidence tier (STRICT / INFERRED / UNKNOWN)
- [x] Model details page shows only applicable metadata (no null-filled
      irrelevant fields — e.g. `onnx_ir_version` never appears for sklearn)
- [x] `POST /api/models/{id}/predict` returns prediction + latency
- [x] `POST /api/families/{family_id}/predict` serves the weighted family alias
- [x] Invalid input returns a typed 400 with a useful message, not a 500
- [x] Every request (success and failure) produces an `inference_logs` row
- [x] Dashboard/monitoring page shows aggregate metrics computed from logs,
      including P95
- [x] Idle/LRU eviction: models beyond `MAX_RESIDENT_MODELS` (default 5)
      are unloaded and re-loaded on next request, with `cold_start` flagged
      in the response
- [x] Model versions, promote/rollback, A/B traffic weights, and input drift
      detection
- [x] No `if framework == ...` branching outside `adapters/`

## Design decisions worth knowing before you extend this

- **Sandbox**: `app/services/sandbox.py` runs pickle-based loads in a
  spawned OS process with blocked outbound sockets (patches
  `socket.socket.__init__`, not the class itself — replacing the class
  breaks anything that subclasses `socket.socket` at import time, like
  `ssl.py`'s `SSLSocket`), `RLIMIT_AS`/`RLIMIT_CPU` limits, and a hard
  wall-clock timeout with `SIGKILL` on overrun. Privilege drop to `nobody`
  is opt-in (`MODELMINT_SANDBOX_DROP_UID=1`) because it requires the
  storage path to already be readable by that uid — wire this up for real
  deployments. This is "raw multiprocessing + resource limits," not
  container-per-load; see the build guide's open decisions (§12) for the
  hardening path if this becomes public-facing.
- **Sandboxed serving (M2)**: `SandboxedAdapterProxy` implements the
  `ModelAdapter` interface but forwards `predict()` over a `multiprocessing`
  pipe to a long-lived worker. The parent only sees JSON-safe prediction
  results. `unload()` (called by LRU eviction) sends shutdown, then
  terminate/kill as a backstop. Per-model requests serialize through a lock,
  which is acceptable for CPU-bound inference; scale out with more workers
  rather than concurrent calls to one model.
- **fork vs spawn**: the sandbox uses `spawn` by default (forking a fresh
  process from inside a live uvicorn/uvloop event loop is a known source of
  hangs) and falls back to `fork` under pytest (spawn's interpreter
  re-exec interacts badly with pytest's assertion-rewrite import hook).
- **joblib, not raw `pickle`**: `joblib.dump()`'s output isn't always a
  plain pickle stream (large numpy arrays get joblib's own wrapper format),
  so the sklearn adapter uses `joblib.load()`, which transparently handles
  both.
- **Swagger**: this is Option A from the build guide (§5) — one generic
  `/predict` route, so `/docs` shows the generic `{"features": [...]}`
  shape for every model. Per-model typed examples live on the Model Details
  page instead (`example_request` field), which does real
  schema-confidence-aware validation server-side. Option B (dynamic route
  registration per model) is intentionally not built — noted as a
  fast-follow.
- **Aggregates are computed on read** from `inference_logs`, not maintained
  as running counters, per the build guide's MVP simplicity call.

## Explicitly out of scope (for now)

Multi-tenancy, billing, autoscaling, GPU scheduling, and a full model registry
with lineage. (API keys, rate limiting, audit logging, model versioning, A/B
testing, and drift detection are now implemented — see the sections above.)

## Project layout

```
PROGRESS.md             live build tracker + architecture decisions

backend/
  alembic/              migrations (0001 baseline, 0002 security, 0003 versions/drift)
  app/
    config.py            pydantic-settings configuration
    cli.py               create/list/revoke API keys
    api/                 models, predict/families, versions, monitoring, keys, meta routes
    security/            auth, rate limits, audit, headers
    models/
      adapters/          ModelAdapter interface + registry.py AdapterSpec catalog
                         (sklearn, onnx, xgboost, lightgbm, catboost, torchscript,
                          keras, tflite, hf, gguf)
      registry.py         Model Manager: LRU cache of resident adapters
      schemas.py           shared pydantic DTOs
    services/
      sandbox.py        subprocess isolation for pickle loads
      sandboxed_adapter.py  persistent sandboxed worker for serving
      analyzer.py        upload -> detect -> load -> validate pipeline
      inference.py        threadpool dispatch for predict()
      api_generator.py     Option A schema validation + example requests
      monitoring.py          aggregate metrics from inference_logs
      drift.py               input summaries + per-feature drift z-scores
    database/         SQLAlchemy models + session (Postgres-ready, SQLite default)
  tests/
    test_adapters/     one fixture model per framework + adapter tests
    test_security.py   API keys, scopes, revocation, rate limits
    test_upload.py     checksums, duplicates, size caps, read auth
    test_m3_formats.py formats, task contracts, batch predict
    test_m5_versions.py versions, promote/rollback, A/B traffic, drift
    test_sandbox_isolation.py
frontend/
  src/
    api/client.js      fetch wrappers + API-key handling
    pages/              Dashboard, ModelDetail, Monitoring, ApiKeys
    components/          StatusChip, Sparkline, KeyModal
```
