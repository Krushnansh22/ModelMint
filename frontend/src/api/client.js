const BASE = "/api";

async function handle(res) {
  if (!res.ok) {
    let detail;
    try {
      detail = await res.json();
    } catch {
      detail = { detail: res.statusText };
    }
    const err = new Error(
      typeof detail.detail === "string"
        ? detail.detail
        : detail.detail?.message || "Request failed"
    );
    err.payload = detail;
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function listModels() {
  return handle(await fetch(`${BASE}/models`));
}

export async function getModel(idOrSlug) {
  return handle(await fetch(`${BASE}/models/${idOrSlug}`));
}

export async function uploadModel(file, name) {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  return handle(await fetch(`${BASE}/models`, { method: "POST", body: form }));
}

export async function predict(idOrSlug, payload) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function getMetrics(modelId) {
  const q = modelId ? `?model_id=${modelId}` : "";
  return handle(await fetch(`${BASE}/monitoring/metrics${q}`));
}

export async function getTimeseries(modelId, limit = 200) {
  const q = new URLSearchParams();
  if (modelId) q.set("model_id", modelId);
  q.set("limit", String(limit));
  return handle(await fetch(`${BASE}/monitoring/timeseries?${q.toString()}`));
}

export async function getLogs(modelId, limit = 50) {
  const q = new URLSearchParams();
  if (modelId) q.set("model_id", modelId);
  q.set("limit", String(limit));
  return handle(await fetch(`${BASE}/monitoring/logs?${q.toString()}`));
}
