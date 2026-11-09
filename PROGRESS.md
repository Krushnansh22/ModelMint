# ModelMint — Build Tracker

> **Purpose:** This file is the single source of truth for the ModelMint build.
> Any new session or agent should read this first, then continue from
> "Current status / next actions". Update the checkboxes and session log as
> work lands.

---

## 1. Project intent

ModelMint is a model-as-a-service platform: upload a trained model artifact,
get it auto-detected, sandbox-validated, stored, and served behind a generic
inference API with monitoring.

The current build adds two major capabilities on top of the original MVP:

1. **Security** — scoped API keys, rate limiting, audit logging, upload
   hardening, security headers, and safe serving of untrusted artifacts.
2. **Multi-format model weights** — serve the most common artifact formats
   (sklearn, ONNX, XGBoost, LightGBM, CatBoost, TorchScript, Keras/TFLite,
   Hugging Face, GGUF) through one task-aware inference contract.

Deployment target: **Docker Compose** with Postgres, MinIO (S3-compatible
storage), Redis (rate limits), and an nginx-served frontend.

---

## 2. Current status

| Field | Value |
|---|---|
| Phase | **M0–M5 complete**; full Compose smoke test pending |
| Last updated | 2026-11-08 |
| Working tree | M0–M5 implemented, 73 tests passing, frontend builds |
| Tests | **73 passed** (`backend/tests`) |
| Frontend | `npm run build` passes |
| Demo | `demo/modelmint-demo.mp4` — 4m35s narrated walkthrough, 24 screenshots, SRT + Markdown transcript |

**Next actions (in order):**
1. Run `docker compose up --build` on a host with Docker Hub access and
   smoke-test upload/predict through MinIO + Postgres (including the `dl` /
   `gguf` targets).
2. Add real fixtures for Keras/TFLite/HF/GGUF once the optional packs are
   installed, and extend `demo/record_demo.py` to cover versioning/traffic/
   drift in the narrated walkthrough.
3. Optional hardening: shared model cache across workers, admin UI accounts,
   PSI/KS drift metrics.

---

## 3. Milestones

### M0 — Foundations
- [x] `backend/app/config.py` using `pydantic-settings`
- [x] Move `DATABASE_URL` / storage / CORS / limits to settings
- [x] Alembic baseline + security migration (`0001_initial`, `0002_security`)
- [x] `init_db()` bootstraps Alembic safely for existing SQLite DBs
- [x] `.env.example`
- [x] `GET /api/formats` (supported formats + optional dependency status)
- [x] Update `requirements.txt`

### M1 — API keys, scopes, rate limits, audit
- [x] `api_keys` table (hash-only storage, prefix lookup, scopes, model binding, expiry, revocation)
- [x] `audit_logs` table
- [x] `backend/app/security/auth.py` — key generation/hash/verify, `AuthContext`, `require_scopes`
- [x] Support `X-API-Key` and `Authorization: Bearer`
- [x] Admin bootstrap via `BOOTSTRAP_ADMIN_KEY`
- [x] `backend/app/cli.py` — create/list/revoke keys
- [x] `backend/app/security/rate_limit.py` — per-key sliding window + per-IP fallback; optional Redis
- [x] `backend/app/security/audit.py` — audit helper
- [x] `backend/app/security/headers.py` — security headers + request ID
- [x] Protect model upload/read/delete, predict, monitoring routes
- [x] `backend/app/api/keys.py` — create/list/revoke/rotate + audit feed
- [x] Upload hardening: streamed write, `MAX_UPLOAD_MB`, SHA-256 checksum
- [x] `DELETE /api/models/{id}` (evict cache + delete artifact + logs)
- [x] Frontend API Keys page + admin unlock + one-time key reveal
- [x] `client.js` attaches key, handles 401/403/429
- [x] Tests: missing key, wrong scope, revoked key, model-bound key, rate limit, hash-only storage,
      upload checksum/duplicate/empty/size cap, audit, configurable read auth

### M2 — Sandboxed serving
- [x] `SandboxedAdapterProxy` with persistent per-model worker
- [x] Serve all `sandboxed` formats through the proxy (no in-process pickle/torch execution)
- [x] Worker cleanup on LRU eviction
- [x] Sandbox tests for inference-time isolation
- [x] Analyzer introspects inside the sandbox (no in-process re-load)
- [x] `INFERENCE_TIMEOUT_SECONDS` per-request timeout

### M3 — Multi-format adapters
- [x] `TaskType` + `InputContract` + task-aware `ModelAdapter`
- [x] `AdapterSpec` registry; derive allowed extensions; `/api/formats`
- [x] Lazy optional dependencies with friendly errors
- [x] XGBoost (`.json`, `.ubj`, `.xgb`)
- [x] LightGBM (`.txt`, `.lgbm`, `.model`)
- [x] CatBoost (`.cbm`)
- [x] TorchScript (`.pt`, `.pth`, `.torchscript`)
- [x] Keras (`.keras`, `.h5`)
- [x] TFLite (`.tflite`)
- [x] Hugging Face bundle (dir/zip)
- [x] GGUF (`.gguf`) — stretch
- [x] Task-aware analyzer, example generator, predict route
- [x] Batch predict endpoint (`POST /api/models/{id}/predict/batch`)
- [x] Frontend task-aware "Try it" panel + batch mode
- [x] Registry/detection/task-aware/batch tests
- [x] `requirements-formats.txt` + `requirements-dl.txt` + `requirements-gguf.txt`
      (slim/full/dl/gguf Docker targets are M4)

> **Note:** the optional format packages are *not* installed in the dev venv,
> so M3 tests cover registry metadata, extension detection,
> optional-dependency error messages, task-aware validation/examples, batch
> predict through the sklearn fixture, and HF zip detection. Add real
> per-format fixtures once the relevant requirements pack is installed.

### M4 — Docker Compose + Postgres + MinIO
- [x] Storage abstraction: `LocalStorage` + `S3Storage` (boto3)
- [x] `Dockerfile` (backend, non-root, slim/full targets)
- [x] `frontend/Dockerfile` (nginx + proxy)
- [x] `docker-compose.yml`: backend, postgres, minio, redis, frontend
- [x] Alembic migrate on startup (`init_db()`; single worker)
- [x] Prometheus `/metrics`, `/health`, `/ready`
- [x] Structured JSON logs (`LOG_JSON`) + request IDs
- [x] GitHub Actions CI (`.github/workflows/ci.yml`)

### M5 — Product polish
- [x] Registry search + framework filter (backend pagination already present;
      no paged UI yet)
- [x] Per-key usage counter (incremented on every predict, shown in key table)
- [x] Copyable curl/Python snippets on model details
- [x] Model versioning / promote / rollback (`0003_versions_drift`,
      `api/versions.py`, Model Details version table)
- [x] A/B traffic split (`/traffic`, `/api/families/{id}/predict` weighted alias)
- [x] Drift detection (per-feature z-scores from JSON input summaries)
- [x] README rewrite

---

## 4. Architecture decisions

- **Only `adapters/registry.py` branches on framework.** Adding a format means
  adding an `AdapterSpec` entry, not editing routes/services.
- **API keys are stored as `HMAC-SHA256(pepper, key)`.** Plaintext is shown
  exactly once at creation/rotation. Lookup uses the full hash; the `prefix`
  column is only for display.
- **Scopes:** `admin`, `models:read`, `models:write`, `predict`,
  `monitoring:read`. `admin` implies all scopes.
- **Read routes are public by default** (`REQUIRE_AUTH_FOR_READS=false`);
  predict is protected by default (`REQUIRE_AUTH_FOR_PREDICT=true`).
- **Rate limits** are in-memory by default (single-process demo). Set
  `REDIS_URL` for a shared limiter across workers.
- **The model cache is per-process.** Run one worker for the demo; a shared
  serving tier is the production path.
- **Untrusted pickle/torch/Keras/HF artifacts must never load in the API
  process.** M2 moves serving into persistent sandboxed workers.
- **Task-aware inference** keeps the original `{"features": [...]}` payload
  working; new contracts are additive (`{"text": ...}`, `{"image_base64": ...}`).
- **Raw `state_dict`/`.safetensors` files are not directly servable** without
  architecture. Support HF bundles (config + weights) or a manifest zip, not
  arbitrary user code by default.

---

## 5. Target API surface

```
GET    /health
GET    /ready
GET    /metrics
GET    /api/formats

POST   /api/keys                 admin
GET    /api/keys                 admin
DELETE /api/keys/{id}            admin
POST   /api/keys/{id}/rotate     admin

POST   /api/models               models:write
GET    /api/models               models:read (or public)
GET    /api/models/{id_or_slug}  models:read (or public)
DELETE /api/models/{id_or_slug}  models:write
POST   /api/models/{id}/predict  predict (+ key model binding)
POST   /api/models/{id}/predict/batch  predict

GET    /api/monitoring/metrics     monitoring:read (or public)
GET    /api/monitoring/timeseries  monitoring:read (or public)
GET    /api/monitoring/logs        monitoring:read (or public)
```

Auth header: `X-API-Key: mm_live_...` or `Authorization: Bearer mm_live_...`.

---

## 6. Data model additions

`api_keys`
`id, name, prefix, key_hash, scopes, model_id, created_at, last_used_at,
expires_at, revoked_at, request_count`

`audit_logs`
`id, actor_key_id, action, resource, status, ip_address, request_id, detail,
timestamp`

`models` additions
`task_type, artifact_format, checksum, storage_backend, storage_key,
framework_version, version, family_id, deleted_at`

`inference_logs` additions
`api_key_id, client_ip, request_id, task_type`

---

## 7. File map (new/changed)

```
PROGRESS.md                                  <- this file

backend/
  .env.example
  alembic.ini
  alembic/env.py
  alembic/script.py.mako
  alembic/versions/0001_initial.py
  alembic/versions/0002_security.py
  app/config.py                              <- new
  app/cli.py                                 <- new
  app/security/__init__.py                   <- new
  app/security/auth.py                       <- new
  app/security/rate_limit.py                 <- new
  app/security/audit.py                      <- new
  app/security/headers.py                    <- new
  app/api/keys.py                            <- new
  app/api/meta.py                            <- new
  app/database/database.py                   <- settings + alembic bootstrap
  app/database/models.py                     <- new tables/columns
  app/api/models.py                          <- auth + upload hardening + delete
  app/api/inference.py                       <- auth + rate limit + key logging
  app/api/monitoring.py                      <- auth
  app/main.py                                <- routers + middleware + bootstrap
  app/models/adapters/registry.py            <- AdapterSpec catalog
  app/models/adapters/{xgboost,lightgbm,catboost,torchscript}_adapter.py
  app/services/sandboxed_adapter.py          <- M2 persistent worker
  app/services/storage.py                    <- M4 Local + S3 backends
  app/services/metrics.py                    <- M4 Prometheus exposition
  tests/test_security.py                     <- new
  tests/test_upload.py                       <- new
  tests/test_m3_formats.py                   <- new
  tests/test_storage.py                      <- new

frontend/
  src/api/client.js                          <- key handling + new endpoints
  src/pages/ApiKeys.jsx                      <- new
  src/components/KeyModal.jsx                <- new
  src/App.jsx                                <- nav + auth status
  src/index.css                              <- styles for key UI

backend/Dockerfile, frontend/Dockerfile, frontend/nginx.conf,
docker-compose.yml, .env.docker.example, .github/workflows/ci.yml
```

---

## 8. Session log

### 2026-11-07 — Session 1
- Read full backend/frontend, identified in-process pickle serving as the
  biggest existing security gap.
- Agreed scope: security + formats first; admin key + scoped keys; Docker
  Compose with Postgres/MinIO; common formats.
- Created this tracker.
- **M0 complete:** `app/config.py`, Alembic (`0001_initial` + `0002_security`),
  safe bootstrap for pre-Alembic DBs, `.env.example`, `/api/formats`,
  `/api/config`, `/ready`, updated requirements.
- **M1 complete (backend):** `api_keys` + `audit_logs`, HMAC key storage,
  scopes, model binding, expiry/revocation, `X-API-Key` + Bearer auth,
  bootstrap admin, CLI, in-memory + Redis sliding-window limiter, audit log,
  security headers + request IDs, protected routes, streamed uploads with
  size cap + SHA-256 + duplicate detection, model delete, per-key inference
  logging.
- **M1 complete (frontend):** API Keys page (admin unlock, create/list/
  rotate/revoke, one-time key modal, audit feed), key-aware `client.js`,
  dynamic upload formats, delete-model action, new styles.
- **Verification:** 22 backend tests pass; `npm run build` passes; full
  bootstrap → create key → upload → predict → delete flow verified against a
  fresh SQLite DB.

### 2026-11-07 — Session 1 (continued) — M2
- Added `app/services/sandboxed_adapter.py`: `SandboxedAdapterProxy` with a
  persistent per-model worker over `multiprocessing.Pipe`.
- Upload-time sandbox now returns JSON-safe `input_schema`, `output_schema`,
  and `metadata`; `analyzer.py` no longer re-loads pickles in the API
  process.
- `ModelManager` serves all sandboxed frameworks through the proxy; LRU
  eviction kills the worker.
- Added `INFERENCE_TIMEOUT_SECONDS` and worker lifecycle tests.
- **Verification:** 23 tests pass; end-to-end upload → predict → delete
  re-verified with the spawn start method (the production path).
- **Next:** M3 multi-format adapters.

### 2026-11-07 — Session 1 (continued) — M3
- Replaced the hand-maintained adapter tables with declarative `AdapterSpec`
  entries (`app/models/adapters/registry.py`); `list_formats()` now reports
  extensions, task types, sandbox flag, optional packages, and installed
  status.
- Added `TaskType` / `InputContract` and task-aware defaults on
  `ModelAdapter` (`build_example_payload`, `predict_batch`).
- New adapters (lazy optional imports, friendly errors when the package is
  missing): XGBoost, LightGBM, CatBoost, TorchScript.
- Analyzer persists `task_type`, `artifact_format`, `framework_version`;
  `api_generator` validates/examples per task (tabular/text/vision/embedding).
- Added `POST /api/models/{id}/predict/batch` (max 1000, per-instance
  validation) and `dispatch_predict_batch`.
- Frontend "Try it" panel is task-aware and has a batch mode toggle.
- Added `backend/requirements-formats.txt`.
- Added `tests/test_m3_formats.py` (registry, detection, task contracts,
  batch predict). Fixed the upload test to use `.csv` now that `.txt` is a
  LightGBM extension.
- **Verification:** 50 tests pass; `npm run build` passes.
- **Next:** M4 Docker Compose + Postgres/MinIO/Redis + storage abstraction.

### 2026-11-07 — Session 1 (continued) — M4
- Added `app/services/storage.py`: `LocalStorage` (atomic move into the
  storage root, path-traversal-safe keys) and `S3Storage` (boto3, MinIO
  compatible, local download cache). Upload streams to a temp file, then hands
  it to the configured backend; `ensure_local_path()` re-downloads S3 objects
  after a restart.
- Added `/metrics` Prometheus exposition (`app/services/metrics.py`), JSON
  logging (`LOG_JSON`), and wired `boto3` into requirements.
- Added `backend/Dockerfile` (non-root, `slim`/`full` targets),
  `frontend/Dockerfile` + `nginx.conf` (SPA + API proxy),
  `docker-compose.yml` (postgres, redis, minio, minio-init, backend,
  frontend), and `.env.docker.example`.
- Added `tests/test_storage.py` (local backend, path safety, metrics).
- Added `.github/workflows/ci.yml` (backend tests + frontend build).
- **Verification:** 54 tests pass; `docker compose config` validates. Docker
  image builds could not complete in this environment because Docker Hub
  metadata requests timed out (TLS handshake) -- rerun `docker compose build`
  on a network with registry access.
- **Next:** full Compose smoke test, then M5 polish.

### 2026-11-07 — Session 1 (continued) — M5 polish
- Incremented `api_keys.request_count` on every predict and exposed it in the
  key-management table.
- Added copyable curl/Python snippets to the model details page.
- Added registry search + framework filter (debounced, uses the existing
  `q`/`framework` query params).
- Rewrote README sections for formats, security, Docker Compose, and
  observability.
- **Verification:** 55 tests pass; `npm run build` passes.
- **Next:** Compose smoke test on a registry-capable host; optional formats.

### 2026-11-08 — Session 2 — demo capture + narrated walkthrough
- Added `demo/record_demo.py`: headless Chrome via CDP walks the full product
  surface, capturing 20 feature screenshots, a screencast, and a narration
  timeline. Google Fonts are blocked so the SPA renders without internet.
- Added `demo/build_narrated_video.py`: offline voiceover via ffmpeg's
  libflite, intro/outro cards, feature-title lower-thirds, burned ASS captions,
  an embedded `mov_text` English subtitle track, and `demo/transcript.md`.
- Covered every feature: registry, scoped keys (locked / unlock / create /
  rotate / revoke), upload + sandbox validation, detected schema, single and
  batch prediction, copy-ready snippets, monitoring, formats API, Prometheus
  metrics, 401 auth rejection, audit log, and model deletion.
- **Verification:** 20 screenshots, 87 screencast frames, 3m49s narrated MP4
  (1440x900, H.264 + AAC + embedded English subtitles), SRT and Markdown
  transcript. `ffmpeg -f null` decode check passes.
- **Note:** the voiceover uses the offline flite TTS voice `slt`. For a human
  VO, replace the generated `demo/audio/*.wav` clips (or the `tts()` call) in
  `demo/build_narrated_video.py` and rerun it.

### 2026-11-08 — Session 3 — optional adapters + M5 governance
- Added four sandboxed adapters and registry specs:
  - `keras` (`.keras`, `.h5`) — shape-driven tabular/vision, no custom objects.
  - `tflite` (`.tflite`) — strict schema from interpreter tensor details.
  - `hf` (`.zip` bundle or directory) — config/tokenizer/weights, pipeline
    task inference, path-traversal-safe extraction, `trust_remote_code=False`.
  - `gguf` (`.gguf`) — llama-cpp-python text generation (stretch).
- Split optional packages into `requirements-formats.txt` (trees + TorchScript),
  `requirements-dl.txt` (TensorFlow/Keras/TFLite/HF), and
  `requirements-gguf.txt`; added `dl` and `gguf` Docker targets.
- Added `0003_versions_drift`: `models.is_active`, `traffic_weight`,
  `promoted_at`, and `inference_logs.model_version` / `input_summary`.
- Uploads now form families (`family_id`, `version`); new versions start
  inactive. Added `POST /api/models/{id}/versions`, promote, rollback, traffic,
  and drift routes, plus `POST /api/families/{id}/predict` for weighted A/B
  serving.
- Added `services/drift.py`: privacy-preserving input summaries and per-feature
  z-score drift with an `INSUFFICIENT_DATA` tier.
- Model Details page: version table, promote/rollback, traffic-weight editor,
  new-version upload, family alias, and drift table.
- **Verification:** 73 backend tests pass; `npm run build` passes; fresh SQLite
  migration to `0003_versions_drift` verified.
- **Next:** Compose smoke test on a registry-capable host; real optional-format
  fixtures; refresh the demo to include governance features.

### 2026-11-08 — Session 4 — narrated demo refresh (M5)
- Extended `demo/record_demo.py` with four governance segments: upload a new
  version, promote it, apply a 50/50 A/B traffic split, and generate a shifted
  request window to show drift (`DRIFT`, z ≈ 10.05).
- Fixed model deletion to drain both family versions so the registry returns
  to its empty state.
- Renumbered the later screenshots and updated `demo/build_narrated_video.py`
  titles/narration (including the intro and outro).
- **Verification:** 24 screenshots, 119 screencast frames, 4m35s narrated MP4
  (1440x900, H.264 + AAC + embedded English subtitles), clean `ffmpeg -f null`
  decode. Updated `demo/README.md` and this tracker.
- **Note:** Swagger UI is still skipped because its CDN assets are blocked in
  the sandbox; the OpenAPI JSON remains available.

---

## 9. How to verify

```bash
# one-time env (repo root)
uv venv --python 3.12 .venv
uv pip install -r backend/requirements.txt --python .venv/bin/python

# backend
cd backend
../.venv/bin/python -m pytest tests/ -q

# frontend
cd frontend
npm install
npm run build

# compose (requires Docker Hub access)
API_KEY_PEPPER=test-pepper docker compose config -q
docker compose up --build
```

For a fresh local DB:
```bash
cd backend
rm -f storage/modelmint.db
python -m uvicorn app.main:app --reload --port 8000
```

Create an admin key:
```bash
cd backend
python -m app.cli create-key --name local-admin --scope admin
```

---

## 10. Known issues / follow-ups

- [x] Existing SQLite DBs need Alembic bootstrap (handled by `init_db()`; verified
      by stamping a pre-Alembic DB and upgrading).
- [ ] Rate limiter is per-process until `REDIS_URL` is set (Compose does).
- [ ] Admin key in `sessionStorage` is demo-grade; real auth needs user accounts.
- [ ] Model cache and worker count are per-process; the backend runs `--workers 1`.
- [x] Untrusted formats are served through persistent sandboxed workers (M2).
- [ ] S3/MinIO artifacts are downloaded into a local cache for serving; a
      shared serving tier with local-disk cache is the production path.
- [ ] Compose stack needs a smoke test on a host with Docker Hub access,
      including the `dl` / `gguf` build targets.
- [x] Keras/TFLite, Hugging Face bundle, and GGUF adapters (optional packs;
      registry/detection tested without the heavy dependencies installed).
- [ ] Versioning is single-family per model; there is no cross-family
      promotion or lineage graph yet.
- [ ] Drift uses mean-shift z-scores only; PSI/KS and per-version baselines are
      follow-ups.
- [x] The narrated demo now covers versions, promotion, traffic splits, and
      drift (4m35s, 24 screenshots).
