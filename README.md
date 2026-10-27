# ModelMint

Upload a trained ML model (sklearn `.pkl`/`.joblib` or ONNX `.onnx`) → it's
auto-detected, sandbox-validated, and served behind a generic inference API
with live monitoring. Built per the accompanying engineering build guide:
sandboxed pickle loading, LRU model eviction, three-tier schema confidence,
and a single generic `/predict` route (Option A from the guide).

Every phase in the build guide's Definition of Done (§11) is implemented and
was exercised end-to-end against real fixture models during development —
see "Verifying it works" below.

## Stack

- **Backend**: FastAPI + SQLAlchemy (SQLite by default, Postgres-ready), scikit-learn, ONNX Runtime
- **Frontend**: React + Vite, no UI framework, plain CSS design tokens
- **Storage**: local filesystem (`backend/storage/`) — swap for S3 later

## Quickstart

### Backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
python -m uvicorn app.main:app --reload --port 8000
```

Runs against a local SQLite file (`backend/storage/modelmint.db`) with zero
setup. To point at Postgres instead:

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

Upload either file through the UI or:

```bash
curl -X POST http://localhost:8000/api/models \
  -F "file=@tests/test_adapters/fixtures/iris_rf.pkl" -F "name=iris_classifier"

curl -X POST http://localhost:8000/api/models/iris-classifier/predict \
  -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
```

### Run tests

```bash
cd backend
python tests/test_adapters/make_fixtures.py   # fixtures are gitignored, generate once
python -m pytest tests/ -q
```

Covers both adapters (detect/load/validate/predict) and sandbox isolation
specifically: a legitimate model loads through the sandbox, a pickle that
tries to open a network socket during unpickling is blocked, and a pickle
that hangs forever is killed on timeout.

## What's implemented (build guide checklist, §11)

- [x] Upload sklearn `.pkl`/`.joblib` and ONNX `.onnx` through the frontend
- [x] Framework/type auto-detected for both (`adapters/registry.py`, the
      only place `if framework == ...` branching is allowed)
- [x] Pickle load happens in a sandboxed subprocess with network egress
      blocked (verified by `test_sandbox_isolation.py`)
- [x] Validation runs synthetic-input test inference, assigns a schema
      confidence tier (STRICT / INFERRED / UNKNOWN)
- [x] Model details page shows only applicable metadata (no null-filled
      irrelevant fields — e.g. `onnx_ir_version` never appears for sklearn)
- [x] `POST /api/models/{id}/predict` returns prediction + latency
- [x] Invalid input returns a typed 400 with a useful message, not a 500
- [x] Every request (success and failure) produces an `inference_logs` row
- [x] Dashboard/monitoring page shows aggregate metrics computed from logs,
      including P95
- [x] Idle/LRU eviction: models beyond `MAX_RESIDENT_MODELS` (default 5)
      are unloaded and re-loaded on next request, with `cold_start` flagged
      in the response
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

## Explicitly out of scope (per the build guide, §1)

Auth, API keys, multi-tenancy, billing, autoscaling, GPU scheduling,
versioning, A/B testing, rate limiting, drift detection.

## Project layout

```
backend/
  app/
    api/            upload/list/get, predict, monitoring routes
    models/
      adapters/      ModelAdapter interface + sklearn/onnx implementations
      registry.py     Model Manager: LRU cache of resident adapters
      schemas.py       shared pydantic DTOs
    services/
      sandbox.py        subprocess isolation for pickle loads
      analyzer.py        upload -> detect -> load -> validate pipeline
      inference.py        threadpool dispatch for predict()
      api_generator.py     Option A schema validation + example requests
      monitoring.py          aggregate metrics from inference_logs
    database/         SQLAlchemy models + session (Postgres-ready, SQLite default)
  tests/
    test_adapters/     one fixture model per framework + adapter tests
    test_sandbox_isolation.py
frontend/
  src/
    api/client.js      fetch wrappers for the backend API
    pages/              Dashboard, ModelDetail, Monitoring
    components/          StatusChip, Sparkline (dependency-free SVG)
```
