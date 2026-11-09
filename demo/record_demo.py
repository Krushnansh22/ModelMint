"""Record the ModelMint UI demo via Chrome DevTools Protocol.

Captures PNG screenshots for every feature, a screencast of the session, and a
timeline of narration events (frame index + text) used by build_narrated_video.py.

Run from the repo root:
    .venv/bin/python demo/record_demo.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "demo")
SHOTS = os.path.join(OUT, "screenshots")
FRAMES = os.path.join(OUT, "frames")
CHROME_PROFILE = "/tmp/opencode/chrome-demo-profile"
ADMIN_KEY = "mm_live_demo_admin_1234567890abcdef"
FIXTURE = os.path.join(
    ROOT, "backend", "tests", "test_adapters", "fixtures", "iris_rf.pkl"
)
APP = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"

for d in (SHOTS, FRAMES):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
shutil.rmtree(CHROME_PROFILE, ignore_errors=True)


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.frames: list[tuple[float, str]] = []
        self.events: list[dict] = []
        self._frame_index = 0
        self._reader_task = None
        self.t0 = time.monotonic()

    async def start(self):
        self._reader_task = asyncio.create_task(self._reader())

    async def _reader(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg:
                fut = self.pending.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg)
                continue
            if msg.get("method") == "Page.screencastFrame":
                params = msg["params"]
                data = base64.b64decode(params["data"])
                ts = time.monotonic()
                name = os.path.join(FRAMES, f"frame_{self._frame_index:05d}.jpg")
                with open(name, "wb") as f:
                    f.write(data)
                self.frames.append((ts, name))
                self._frame_index += 1
                asyncio.create_task(
                    self.call("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
                )

    async def call(self, method, params=None, timeout=30):
        self._id += 1
        msg_id = self._id
        fut = asyncio.get_event_loop().create_future()
        self.pending[msg_id] = fut
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        msg = await asyncio.wait_for(fut, timeout=timeout)
        if "error" in msg:
            raise RuntimeError(f"{method} failed: {msg['error']}")
        return msg.get("result", {})

    async def js(self, expression):
        res = await self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(res["exceptionDetails"])
        return res.get("result", {}).get("value")

    async def wait_for(self, expression, timeout=30, label=""):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                # Coerce to boolean so DOM nodes (not serializable by value)
                # still report as "present".
                if await self.js(f"!!({expression})"):
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.15)
        raise TimeoutError(f"timed out waiting for {label or expression}")

    async def screenshot(self, name):
        await asyncio.sleep(0.35)
        res = await self.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        path = os.path.join(SHOTS, name)
        with open(path, "wb") as f:
            f.write(base64.b64decode(res["data"]))
        print(f"  screenshot -> {path}", flush=True)

    def mark(self, name, narration):
        self.events.append(
            {
                "name": name,
                "frame_index": len(self.frames),
                "t": round(time.monotonic() - self.t0, 3),
                "narration": narration,
            }
        )
        print(f"  narration  -> {name}", flush=True)


async def click_button(cdp: CDP, text: str):
    expr = (
        "(() => { const b=[...document.querySelectorAll('button')]"
        f".find(x => x.textContent.trim().toLowerCase().includes({json.dumps(text.lower())}));"
        " if (!b) return false; b.click(); return true; })()"
    )
    if not await cdp.js(expr):
        raise RuntimeError(f"button not found: {text}")
    await asyncio.sleep(0.25)


async def set_input(cdp: CDP, selector: str, value: str):
    expr = f"""
    (() => {{
      const el = document.querySelector({json.dumps(selector)});
      if (!el) return false;
      const proto = el instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
      setter.call(el, {json.dumps(value)});
      el.dispatchEvent(new Event('input', {{ bubbles: true }}));
      return true;
    }})()
    """
    if not await cdp.js(expr):
        raise RuntimeError(f"input not found: {selector}")
    await asyncio.sleep(0.2)


async def click_selector(cdp: CDP, selector: str):
    expr = f"(() => {{ const el=document.querySelector({json.dumps(selector)}); if(!el) return false; el.click(); return true; }})()"
    if not await cdp.js(expr):
        raise RuntimeError(f"selector not found: {selector}")
    await asyncio.sleep(0.25)


async def click_key_row_button(cdp: CDP, row_name: str, button_text: str):
    expr = f"""(() => {{
      const row = [...document.querySelectorAll('.key-row')]
        .find(r => r.textContent.includes({json.dumps(row_name)}));
      if (!row) return false;
      const b = [...row.querySelectorAll('button')]
        .find(x => x.textContent.trim().toLowerCase() === {json.dumps(button_text.lower())});
      if (!b) return false;
      b.click();
      return true;
    }})()"""
    if not await cdp.js(expr):
        raise RuntimeError(f"row button not found: {row_name}/{button_text}")
    await asyncio.sleep(0.3)


async def scroll_to(cdp: CDP, selector: str):
    await cdp.js(
        f"(() => {{ const el=document.querySelector({json.dumps(selector)});"
        " if (el) el.scrollIntoView({block:'center'}); return true; })()"
    )
    await asyncio.sleep(0.4)


async def navigate(cdp: CDP, url: str, wait_expr: str | None = None, timeout=30):
    await cdp.call("Page.navigate", {"url": url})
    await asyncio.sleep(0.8)
    if wait_expr:
        await cdp.wait_for(wait_expr, timeout=timeout, label=url)


def _api_get(path: str) -> str:
    req = urllib.request.Request(API + path, headers={"X-API-Key": ADMIN_KEY})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode()


def _api_post(path: str, payload: dict) -> str:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"X-API-Key": ADMIN_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode()


async def main():
    chrome = subprocess.Popen(
        [
            "/usr/bin/google-chrome",
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            "--remote-debugging-port=9222",
            "--user-data-dir=" + CHROME_PROFILE,
            "--window-size=1440,960",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        target = None
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
                    targets = json.load(r)
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    target = pages[0]
                    break
            except Exception:
                pass
            await asyncio.sleep(0.25)
        if target is None:
            raise RuntimeError("Chrome debugging endpoint did not come up")

        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            await cdp.start()
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("DOM.enable")
            await cdp.call("Network.enable")
            # The sandbox has no internet; Google Fonts would block module
            # execution until the request times out. Block it up front.
            await cdp.call(
                "Network.setBlockedURLs",
                {"urls": ["*fonts.googleapis.com*", "*fonts.gstatic.com*"]},
            )
            await cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
            )
            await cdp.call(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 80,
                    "maxWidth": 1440,
                    "maxHeight": 900,
                    "everyNthFrame": 1,
                },
            )
            cdp.t0 = time.monotonic()

            print("1. dashboard", flush=True)
            await navigate(cdp, APP, "document.querySelector('.page-title')")
            await cdp.wait_for(
                "document.body.innerText.toLowerCase().includes('no models minted yet')", 20
            )
            await cdp.screenshot("01-dashboard-empty.png")
            cdp.mark(
                "01-dashboard-empty",
                "This is ModelMint, a model as a service platform. The registry is empty, "
                "ready for a first model upload.",
            )

            print("2. api keys locked", flush=True)
            await click_button(cdp, "API keys")
            await cdp.wait_for("document.querySelector('.key-unlock')")
            await cdp.screenshot("02-api-keys-locked.png")
            cdp.mark(
                "02-api-keys-locked",
                "The API keys page starts locked. Administrative actions require an admin key.",
            )

            print("3. unlock admin", flush=True)
            await set_input(cdp, ".key-unlock input", ADMIN_KEY)
            await click_button(cdp, "Unlock")
            await cdp.wait_for(
                "document.body.innerText.toLowerCase().includes('create a key')", 20
            )
            await cdp.screenshot("03-api-keys-unlocked.png")
            cdp.mark(
                "03-api-keys-unlocked",
                "Unlocking with the bootstrap admin key reveals scoped key management "
                "and the security audit log.",
            )

            print("4. create scoped key", flush=True)
            await set_input(cdp, ".key-form input", "demo-client")
            await set_input(cdp, ".key-form input[type=number]", "30")
            for scope in ("models:write", "models:read", "monitoring:read"):
                await cdp.js(
                    "(() => { const l=[...document.querySelectorAll('.scope-toggle')]"
                    f".find(x => x.textContent.trim() === {json.dumps(scope)});"
                    " if (l) l.click(); return !!l; })()"
                )
                await asyncio.sleep(0.15)
            await click_button(cdp, "Create key")
            await cdp.wait_for("document.querySelector('.modal')", 20)
            await cdp.screenshot("04-key-created-modal.png")
            cdp.mark(
                "04-key-created-modal",
                "Create a scoped key for a client. The plaintext secret is shown exactly "
                "once, then stored only as a hash.",
            )
            await click_button(cdp, "Done")
            await asyncio.sleep(0.5)
            await cdp.screenshot("05-keys-and-audit.png")
            cdp.mark(
                "05-keys-and-audit",
                "The client key has read, write, and monitoring scopes, and the audit log "
                "records the key creation.",
            )

            print("5. upload model", flush=True)
            await click_button(cdp, "Registry")
            await cdp.wait_for("document.querySelector('.upload-panel')")
            res = await cdp.call(
                "Runtime.evaluate", {"expression": "document.querySelector('input[type=file]')"}
            )
            object_id = res["result"]["objectId"]
            await cdp.call("DOM.setFileInputFiles", {"files": [FIXTURE], "objectId": object_id})
            await cdp.wait_for(
                "document.querySelectorAll('.ledger-row:not(.header)').length > 0 "
                "&& (document.querySelector('.ledger-row:not(.header) .chip')?.textContent || '')"
                ".toLowerCase().includes('ready')",
                timeout=90,
                label="model READY",
            )
            await cdp.screenshot("06-registry-uploaded.png")
            cdp.mark(
                "06-registry-uploaded",
                "Uploading a scikit-learn pickle is detected, sandbox validated, checksummed, "
                "and given a live inference endpoint automatically.",
            )

            print("6. model detail", flush=True)
            await click_selector(cdp, ".ledger-row:not(.header)")
            await cdp.wait_for("document.querySelector('.detail-title')")
            await cdp.screenshot("07-model-detail.png")
            cdp.mark(
                "07-model-detail",
                "Model details show the detected framework, task type, schema confidence, "
                "feature count, and load time.",
            )

            print("7. single predict", flush=True)
            await scroll_to(cdp, ".tryit")
            await cdp.screenshot("08-try-it.png")
            cdp.mark(
                "08-try-it",
                "The Try It panel generates a task aware example request for this tabular model.",
            )
            await click_button(cdp, "Send request")
            await cdp.wait_for("document.querySelector('.result-block pre')", 30)
            await cdp.screenshot("09-predict-result.png")
            cdp.mark(
                "09-predict-result",
                "A single prediction returns in milliseconds from the persistent sandboxed worker.",
            )

            print("8. batch predict", flush=True)
            await click_selector(cdp, ".batch-toggle input")
            await set_input(
                cdp,
                ".tryit textarea",
                json.dumps(
                    [
                        {"features": [5.1, 3.5, 1.4, 0.2]},
                        {"features": [6.2, 3.4, 5.4, 2.3]},
                    ]
                ),
            )
            await click_button(cdp, "Send request")
            await cdp.wait_for(
                "document.querySelector('.result-block pre')?.textContent.includes('[')", 30
            )
            await cdp.screenshot("10-batch-result.png")
            cdp.mark(
                "10-batch-result",
                "Batch mode accepts an array of instances and returns one prediction per row.",
            )

            print("9. request examples", flush=True)
            await scroll_to(cdp, ".snippet")
            await cdp.screenshot("11-request-examples.png")
            cdp.mark(
                "11-request-examples",
                "Copy ready curl and Python snippets make integration immediate.",
            )

            print("10. monitoring", flush=True)
            await click_button(cdp, "Monitoring")
            await cdp.wait_for("document.querySelector('.stat-row')", 20)
            await asyncio.sleep(0.4)
            await cdp.screenshot("12-monitoring.png")
            cdp.mark(
                "12-monitoring",
                "Monitoring aggregates request counts, success rate, latency percentiles, "
                "and recent inference logs.",
            )

            print("11. versions and promotion", flush=True)
            await click_button(cdp, "Registry")
            await cdp.wait_for("document.querySelector('.ledger-row:not(.header)')", 20)
            await click_selector(cdp, ".ledger-row:not(.header)")
            await cdp.wait_for("document.querySelector('.version-table')", 20)

            # Upload a second version through the hidden file input.
            res = await cdp.call(
                "Runtime.evaluate", {"expression": "document.querySelector('.hidden-input')"}
            )
            object_id = res["result"]["objectId"]
            await cdp.call(
                "DOM.setFileInputFiles", {"files": [FIXTURE], "objectId": object_id}
            )
            await cdp.wait_for(
                "document.querySelectorAll('.version-row:not(.header)').length >= 2",
                120,
                label="second version",
            )
            await scroll_to(cdp, ".version-table")
            await cdp.screenshot("13-version-uploaded.png")
            cdp.mark(
                "13-version-uploaded",
                "Upload a new version into the same model family. It is sandbox validated "
                "and starts inactive, so production traffic is untouched until promotion.",
            )

            # Promote the newest version (top row).
            await cdp.js(
                """(() => {
                  const row = document.querySelectorAll('.version-row:not(.header)')[0];
                  const b = [...row.querySelectorAll('button')]
                    .find(x => x.textContent.trim() === 'promote');
                  if (!b) return false;
                  b.click();
                  return true;
                })()"""
            )
            await cdp.wait_for(
                "(() => { const rows=document.querySelectorAll('.version-row:not(.header)');"
                " return rows.length >= 2 && !!rows[0].querySelector('.active-badge'); })()",
                30,
                label="v2 active",
            )
            await cdp.screenshot("14-version-promoted.png")
            cdp.mark(
                "14-version-promoted",
                "Promoting version two makes it the active release. Version one stays "
                "available for an instant rollback.",
            )

            # Apply a 50/50 traffic split.
            await cdp.js(
                """(() => {
                  const inputs = [...document.querySelectorAll('.weight-input')];
                  const values = ['0.5', '0.5'];
                  inputs.forEach((el, i) => {
                    const setter = Object.getOwnPropertyDescriptor(
                      HTMLInputElement.prototype, 'value').set;
                    setter.call(el, values[i] ?? '0');
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                  });
                  return inputs.length;
                })()"""
            )
            await click_button(cdp, "Apply traffic split")
            await cdp.wait_for(
                "document.body.innerText.includes('Traffic split applied')",
                30,
                label="traffic applied",
            )
            await cdp.screenshot("15-traffic-split.png")
            cdp.mark(
                "15-traffic-split",
                "Traffic weights let you canary a release. Here version two and version one "
                "each receive fifty percent of requests through the family endpoint.",
            )

            print("12. drift detection", flush=True)
            # Generate a baseline and a shifted window against the newest version,
            # then reopen its detail page so the drift panel refetches.
            models = json.loads(await asyncio.to_thread(_api_get, "/api/models"))
            model_id = models[0]["id"]
            # Baseline with realistic spread, then a shifted current window.
            for value in (-1.5, -1.2, -0.9, -0.6, -0.3, 0.3, 0.6, 0.9, 1.2, 1.5):
                await asyncio.to_thread(
                    _api_post,
                    f"/api/models/{model_id}/predict",
                    {"features": [value, 0.0, 0.0, 0.0]},
                )
            for _ in range(10):
                await asyncio.to_thread(
                    _api_post,
                    f"/api/models/{model_id}/predict",
                    {"features": [10.0, 0.0, 0.0, 0.0]},
                )
            await click_button(cdp, "Registry")
            await cdp.wait_for("document.querySelector('.ledger-row:not(.header)')", 20)
            await click_selector(cdp, ".ledger-row:not(.header)")
            await cdp.wait_for("document.querySelector('.drift-status')", 30)
            await scroll_to(cdp, ".drift-table")
            await cdp.screenshot("16-drift.png")
            cdp.mark(
                "16-drift",
                "Drift detection compares the recent request window with the baseline and "
                "flags features whose distribution has shifted, without storing raw inputs.",
            )

            print("13. formats json", flush=True)
            await navigate(cdp, f"{API}/api/formats", "document.body")
            await cdp.screenshot("17-formats-api.png")
            cdp.mark(
                "17-formats-api",
                "The formats endpoint reports every supported framework and whether its "
                "optional package is installed.",
            )

            print("14. metrics", flush=True)
            await navigate(cdp, f"{API}/metrics", "document.body")
            await cdp.screenshot("18-prometheus-metrics.png")
            cdp.mark(
                "18-prometheus-metrics",
                "Prometheus metrics expose model counts, request totals, latency, and "
                "active API keys.",
            )

            print("15. auth required", flush=True)
            await navigate(cdp, APP, "document.querySelector('.page-title')")
            await cdp.js(
                """(async () => {
                  const r = await fetch('http://127.0.0.1:8000/api/models/iris-rf/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: [5.1, 3.5, 1.4, 0.2] }),
                  });
                  const t = await r.text();
                  document.body.innerHTML =
                    '<pre style="font:14px monospace;padding:28px;white-space:pre-wrap">HTTP ' +
                    r.status + ' ' + r.statusText + '\\n\\n' +
                    t.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</pre>';
                  return r.status;
                })()"""
            )
            await asyncio.sleep(0.5)
            await cdp.screenshot("19-auth-required.png")
            cdp.mark(
                "19-auth-required",
                "Calling predict without an API key is rejected with 401 Unauthorized.",
            )

            print("16. openapi docs", flush=True)
            await navigate(cdp, f"{API}/docs", "document.body")
            await asyncio.sleep(2.5)
            if await cdp.js("!!document.querySelector('.swagger-ui')"):
                await cdp.screenshot("19b-api-docs.png")
                cdp.mark(
                    "19b-api-docs",
                    "Interactive OpenAPI documentation is available at the docs endpoint.",
                )
            else:
                print("  swagger UI unavailable (CDN blocked); skipped", flush=True)

            print("17. audit log", flush=True)
            await navigate(cdp, APP, "document.querySelector('.page-title')")
            await click_button(cdp, "API keys")
            await cdp.wait_for(
                "document.body.innerText.toLowerCase().includes('create a key')", 20
            )
            await scroll_to(cdp, ".audit-list")
            await cdp.screenshot("20-audit-log.png")
            cdp.mark(
                "20-audit-log",
                "The audit log tracks key creation, uploads, promotions, and other "
                "security events.",
            )

            print("18. rotate key", flush=True)
            await cdp.js("window.confirm = () => true")
            await click_key_row_button(cdp, "demo-client", "rotate")
            await cdp.wait_for("document.querySelector('.modal')", 20)
            await cdp.screenshot("21-key-rotated.png")
            cdp.mark(
                "21-key-rotated",
                "Rotating a key immediately invalidates the old secret and issues a new one.",
            )
            await click_button(cdp, "Done")
            await asyncio.sleep(0.4)

            print("19. revoke key", flush=True)
            await cdp.js("window.confirm = () => true")
            await click_key_row_button(cdp, "demo-client", "revoke")
            await cdp.wait_for(
                "[...document.querySelectorAll('.key-row')].some(r => "
                "r.textContent.includes('demo-client') && "
                "r.textContent.toLowerCase().includes('revoked'))",
                20,
            )
            await cdp.screenshot("22-key-revoked.png")
            cdp.mark(
                "22-key-revoked",
                "Revoking a key disables it permanently while preserving the audit trail.",
            )

            print("20. delete model", flush=True)
            # The family now has two versions; delete each until the registry is empty.
            for _ in range(5):
                await click_button(cdp, "Registry")
                await cdp.wait_for("document.querySelector('.ledger-row:not(.header)')", 20)
                await click_selector(cdp, ".ledger-row:not(.header)")
                await cdp.wait_for("document.querySelector('.detail-title')", 20)
                await cdp.js("window.confirm = () => true")
                await click_button(cdp, "Delete model")
                await asyncio.sleep(1.0)
                if await cdp.js(
                    "document.body.innerText.toLowerCase().includes('no models minted yet')"
                ):
                    break
            await cdp.wait_for(
                "document.body.innerText.toLowerCase().includes('no models minted yet')", 30
            )
            await cdp.screenshot("23-model-deleted.png")
            cdp.mark(
                "23-model-deleted",
                "Deleting a model removes its artifact and endpoint, returning the registry "
                "to an empty state.",
            )

            print("21. final audit", flush=True)
            await click_button(cdp, "API keys")
            await cdp.wait_for(
                "document.body.innerText.toLowerCase().includes('create a key')", 20
            )
            await scroll_to(cdp, ".audit-list")
            await cdp.screenshot("24-final-audit.png")
            cdp.mark(
                "24-final-audit",
                "Every security sensitive action remains visible in the audit trail. "
                "ModelMint is ready to mint, serve, and govern models.",
            )

            await cdp.call("Page.stopScreencast")
            await asyncio.sleep(0.5)
            frames = list(cdp.frames)
            events = list(cdp.events)
            await ws.close()

        print(f"captured {len(frames)} screencast frames and {len(events)} events", flush=True)

        meta = {
            "frames": [
                {"path": os.path.relpath(path, OUT), "t": round(ts - cdp.t0, 3)}
                for ts, path in frames
            ],
            "events": events,
        }
        with open(os.path.join(OUT, "timeline.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"timeline -> {os.path.join(OUT, 'timeline.json')}", flush=True)

        write_silent_video(frames)
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:
            chrome.kill()


def write_silent_video(frames):
    if len(frames) < 2:
        print("not enough frames for a video")
        return
    concat = os.path.join(OUT, "frames.txt")
    with open(concat, "w") as f:
        for i, (ts, path) in enumerate(frames):
            duration = frames[i + 1][0] - ts if i + 1 < len(frames) else 1.5
            duration = max(0.05, min(duration, 5.0))
            f.write(f"file '{path}'\n")
            f.write(f"duration {duration:.3f}\n")
        f.write(f"file '{frames[-1][1]}'\n")

    out = os.path.join(OUT, "modelmint-demo-screencast.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat,
        "-vf", "fps=20,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-movflags", "+faststart", out,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"screencast -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
