const BASE = "/api";
const KEY_STORAGE = "modelmint.apiKey";

export function getApiKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export function setApiKey(key) {
  sessionStorage.setItem(KEY_STORAGE, key.trim());
}

export function clearApiKey() {
  sessionStorage.removeItem(KEY_STORAGE);
}

function authHeaders(extra = {}) {
  const key = getApiKey();
  return key ? { ...extra, "X-API-Key": key } : extra;
}

async function handle(res) {
  if (!res.ok) {
    let detail;
    try {
      detail = await res.json();
    } catch {
      detail = { detail: res.statusText };
    }
    const message =
      typeof detail.detail === "string"
        ? detail.detail
        : detail.detail?.message || detail.message || "Request failed";
    const err = new Error(message);
    err.payload = detail;
    err.status = res.status;
    if (res.status === 401) {
      err.message = "Missing or invalid API key. Unlock the API Keys page.";
    }
    if (res.status === 403) {
      err.message = detail.detail?.message || "This API key lacks the required scope.";
    }
    if (res.status === 429) {
      err.message = "Rate limit exceeded. Try again shortly.";
    }
    throw err;
  }
  return res.json();
}

export async function listModels(params = {}) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") q.set(k, v);
  });
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return handle(await fetch(`${BASE}/models${suffix}`, { headers: authHeaders() }));
}

export async function getModel(idOrSlug) {
  return handle(await fetch(`${BASE}/models/${idOrSlug}`, { headers: authHeaders() }));
}

export async function uploadModel(file, name, family) {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  if (family) form.append("family", family);
  return handle(
    await fetch(`${BASE}/models`, { method: "POST", body: form, headers: authHeaders() })
  );
}

export async function deleteModel(idOrSlug) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}`, { method: "DELETE", headers: authHeaders() })
  );
}

export async function predict(idOrSlug, payload) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/predict`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    })
  );
}

export async function predictBatch(idOrSlug, instances) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/predict/batch`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ instances }),
    })
  );
}

export async function getMetrics(modelId) {
  const q = modelId ? `?model_id=${modelId}` : "";
  return handle(await fetch(`${BASE}/monitoring/metrics${q}`, { headers: authHeaders() }));
}

export async function getTimeseries(modelId, limit = 200) {
  const q = new URLSearchParams();
  if (modelId) q.set("model_id", modelId);
  q.set("limit", String(limit));
  return handle(
    await fetch(`${BASE}/monitoring/timeseries?${q.toString()}`, { headers: authHeaders() })
  );
}

export async function getLogs(modelId, limit = 50) {
  const q = new URLSearchParams();
  if (modelId) q.set("model_id", modelId);
  q.set("limit", String(limit));
  return handle(
    await fetch(`${BASE}/monitoring/logs?${q.toString()}`, { headers: authHeaders() })
  );
}

export async function listFormats() {
  return handle(await fetch(`${BASE}/formats`));
}

// --- versioning / A/B / drift (M5) ---------------------------------------
export async function listVersions(idOrSlug) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/versions`, { headers: authHeaders() })
  );
}

export async function uploadVersion(idOrSlug, file, name) {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/versions`, {
      method: "POST",
      body: form,
      headers: authHeaders(),
    })
  );
}

export async function promoteVersion(idOrSlug) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/promote`, {
      method: "POST",
      headers: authHeaders(),
    })
  );
}

export async function rollbackVersion(idOrSlug) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/rollback`, {
      method: "POST",
      headers: authHeaders(),
    })
  );
}

export async function setTraffic(idOrSlug, weights) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/traffic`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ weights }),
    })
  );
}

export async function getDrift(idOrSlug, window = 100) {
  return handle(
    await fetch(`${BASE}/models/${idOrSlug}/drift?window=${window}`, {
      headers: authHeaders(),
    })
  );
}

export async function predictFamily(familyId, payload) {
  return handle(
    await fetch(`${BASE}/families/${familyId}/predict`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    })
  );
}

export async function getPublicConfig() {
  return handle(await fetch(`${BASE}/config`));
}

// --- API key management (admin scope) ------------------------------------
export async function listKeys() {
  return handle(await fetch(`${BASE}/keys`, { headers: authHeaders() }));
}

export async function createKey(payload) {
  return handle(
    await fetch(`${BASE}/keys`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    })
  );
}

export async function revokeKey(id) {
  return handle(
    await fetch(`${BASE}/keys/${id}`, { method: "DELETE", headers: authHeaders() })
  );
}

export async function rotateKey(id) {
  return handle(
    await fetch(`${BASE}/keys/${id}/rotate`, { method: "POST", headers: authHeaders() })
  );
}

export async function listAuditLogs(limit = 100) {
  return handle(
    await fetch(`${BASE}/keys/audit?limit=${limit}`, { headers: authHeaders() })
  );
}
