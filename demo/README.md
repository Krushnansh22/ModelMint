# ModelMint Demo

A recorded, narrated walkthrough of the full ModelMint product surface.

## Watch / read

| File | What it is |
|---|---|
| `modelmint-demo.mp4` | 4m35s narrated walkthrough, 1440x900, H.264/AAC, burned lower-third captions, embedded English subtitle track |
| `modelmint-demo-screencast.mp4` | Silent raw screencast of the same session |
| `transcript.md` | Timestamped, readable transcript of every segment |
| `modelmint-demo.srt` | Subtitle / transcript file |

## Screenshots

`demo/screenshots/` has 24 PNGs, one per feature:

| # | File | Feature |
|---|---|---|
| 01 | `01-dashboard-empty.png` | Empty model registry |
| 02 | `02-api-keys-locked.png` | API keys page locked |
| 03 | `03-api-keys-unlocked.png` | Admin unlock |
| 04 | `04-key-created-modal.png` | One-time scoped key secret |
| 05 | `05-keys-and-audit.png` | Key scopes + audit feed |
| 06 | `06-registry-uploaded.png` | Upload + sandbox validation |
| 07 | `07-model-detail.png` | Detected schema / metadata |
| 08 | `08-try-it.png` | Task-aware Try It |
| 09 | `09-predict-result.png` | Single prediction |
| 10 | `10-batch-result.png` | Batch prediction |
| 11 | `11-request-examples.png` | curl / Python snippets |
| 12 | `12-monitoring.png` | Monitoring + latency |
| 13 | `13-version-uploaded.png` | Upload a new model version |
| 14 | `14-version-promoted.png` | Promote a release |
| 15 | `15-traffic-split.png` | A/B traffic weights |
| 16 | `16-drift.png` | Input drift detection |
| 17 | `17-formats-api.png` | Supported formats API |
| 18 | `18-prometheus-metrics.png` | Prometheus metrics |
| 19 | `19-auth-required.png` | 401 without an API key |
| 20 | `20-audit-log.png` | Security audit log |
| 21 | `21-key-rotated.png` | Key rotation |
| 22 | `22-key-revoked.png` | Key revocation |
| 23 | `23-model-deleted.png` | Model deletion |
| 24 | `24-final-audit.png` | Final audit trail |

## Regenerate

Start the demo stack from the repo root:

```bash
cd backend
rm -f storage/demo.db
rm -rf storage/_uploads storage/_cache
find storage -maxdepth 1 -type f ! -name '.gitkeep' -delete

DATABASE_URL=sqlite:///./storage/demo.db \
API_KEY_PEPPER=demo-pepper \
BOOTSTRAP_ADMIN_KEY=mm_live_demo_admin_1234567890abcdef \
ENVIRONMENT=development \
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Then, from the repo root:

```bash
.venv/bin/python demo/record_demo.py
.venv/bin/python demo/build_narrated_video.py
```

`record_demo.py` uses headless Chrome via the DevTools Protocol and writes
`timeline.json`, `screenshots/`, `frames/`, and the silent screencast.
`build_narrated_video.py` synthesizes the voiceover, renders the intro/outro
cards and lower-thirds, and muxes the final MP4, SRT, and transcript.

## Notes

- The voiceover is generated offline with ffmpeg's built-in `libflite` TTS
  (`slt` voice). To use a human recording, replace the generated
  `demo/audio/*.wav` files and rerun `build_narrated_video.py`, or swap the
  `tts()` implementation.
- The recorder blocks `fonts.googleapis.com` and `fonts.gstatic.com` so the
  single-page app renders in environments without internet access.
- Requirements: `/usr/bin/google-chrome`, `ffmpeg` built with `libass` and
  `flite`, and the repo `.venv` with `websockets`.
