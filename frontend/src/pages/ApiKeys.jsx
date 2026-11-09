import { useCallback, useEffect, useState } from "react";
import {
  clearApiKey,
  createKey,
  getApiKey,
  listAuditLogs,
  listKeys,
  listModels,
  revokeKey,
  rotateKey,
  setApiKey,
} from "../api/client.js";
import KeyModal from "../components/KeyModal.jsx";

const SCOPES = ["predict", "models:read", "models:write", "monitoring:read", "admin"];

function fmtDate(value) {
  return value ? new Date(value).toLocaleString() : "—";
}

export default function ApiKeys() {
  const [unlocked, setUnlocked] = useState(Boolean(getApiKey()));
  const [keyInput, setKeyInput] = useState("");
  const [keys, setKeys] = useState([]);
  const [models, setModels] = useState([]);
  const [audit, setAudit] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState(null);

  const [name, setName] = useState("");
  const [scopes, setScopes] = useState(["predict"]);
  const [modelId, setModelId] = useState("");
  const [expiresDays, setExpiresDays] = useState("");

  const refresh = useCallback(async () => {
    const [keyRows, modelRows, auditRows] = await Promise.all([
      listKeys(),
      listModels(),
      listAuditLogs(50),
    ]);
    setKeys(keyRows);
    setModels(modelRows);
    setAudit(auditRows);
    setError(null);
  }, []);

  useEffect(() => {
    if (!unlocked) return;
    refresh().catch((e) => {
      if (e.status === 401) {
        clearApiKey();
        setUnlocked(false);
      }
      setError(e.message);
    });
  }, [unlocked, refresh]);

  async function unlock() {
    setError(null);
    setBusy(true);
    try {
      setApiKey(keyInput);
      await refresh();
      setUnlocked(true);
      setKeyInput("");
    } catch (e) {
      clearApiKey();
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function lock() {
    clearApiKey();
    setUnlocked(false);
    setKeys([]);
    setAudit([]);
  }

  function toggleScope(scope) {
    setScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  }

  async function submitCreate(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const body = {
        name: name.trim(),
        scopes,
        model_id: modelId || null,
        expires_in_days: expiresDays ? Number(expiresDays) : null,
      };
      const row = await createKey(body);
      setCreated(row);
      setName("");
      setExpiresDays("");
      await refresh();
    } catch (e2) {
      setError(e2.message);
    } finally {
      setBusy(false);
    }
  }

  async function onRevoke(row) {
    if (!window.confirm(`Revoke "${row.name}"? This cannot be undone.`)) return;
    setError(null);
    try {
      await revokeKey(row.id);
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  }

  async function onRotate(row) {
    if (!window.confirm(`Rotate "${row.name}"? The old key stops working immediately.`)) return;
    setError(null);
    try {
      const fresh = await rotateKey(row.id);
      setCreated(fresh);
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  }

  if (!unlocked) {
    return (
      <div>
        <div className="page-head">
          <h1 className="page-title">API keys</h1>
          <p className="page-desc">
            Paste an admin key to manage scoped keys. The key is kept in this browser tab only.
          </p>
        </div>

        <div className="section key-unlock">
          <h2 className="section-title">Admin unlock</h2>
          <p className="key-hint">
            Create one with{" "}
            <code>python -m app.cli create-key --name admin --scope admin</code>, or set
            <code> BOOTSTRAP_ADMIN_KEY</code> before starting the backend.
          </p>
          <div className="key-form-row">
            <input
              type="password"
              value={keyInput}
              placeholder="mm_live_…"
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && keyInput.trim() && unlock()}
            />
            <button className="btn" disabled={busy || !keyInput.trim()} onClick={unlock}>
              {busy ? "Checking…" : "Unlock"}
            </button>
          </div>
          {error && <p className="error-banner">{error}</p>}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">API keys</h1>
        <p className="page-desc">
          Scoped keys for uploads, inference, and monitoring. Keys are stored hashed; the
          plaintext is shown once.
        </p>
      </div>

      <div className="keys-toolbar">
        <span className="chip ready">admin unlocked</span>
        <button className="btn secondary" onClick={lock}>
          Lock
        </button>
      </div>

      {error && <p className="error-banner">{error}</p>}

      <div className="section">
        <h2 className="section-title">Create a key</h2>
        <form className="key-form" onSubmit={submitCreate}>
          <div className="key-form-row">
            <label className="key-label">
              Name
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="ci-pipeline"
                required
              />
            </label>
            <label className="key-label">
              Expires in (days, optional)
              <input
                type="number"
                min="1"
                value={expiresDays}
                onChange={(e) => setExpiresDays(e.target.value)}
                placeholder="90"
              />
            </label>
            <label className="key-label">
              Bind to model (optional)
              <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                <option value="">any model</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="scope-row">
            {SCOPES.map((scope) => (
              <label key={scope} className={`scope-toggle ${scopes.includes(scope) ? "on" : ""}`}>
                <input
                  type="checkbox"
                  checked={scopes.includes(scope)}
                  onChange={() => toggleScope(scope)}
                />
                {scope}
              </label>
            ))}
          </div>

          <button className="btn" disabled={busy || !name.trim() || scopes.length === 0}>
            {busy ? "Creating…" : "Create key"}
          </button>
        </form>
      </div>

      <div className="section">
        <h2 className="section-title">Keys</h2>
        {keys.length === 0 ? (
          <div className="empty-state">No keys yet.</div>
        ) : (
          <div className="key-table">
            <div className="key-row header">
              <span>Name</span>
              <span>Prefix</span>
              <span>Scopes</span>
              <span>Binding</span>
              <span>Requests</span>
              <span>Last used</span>
              <span>Status</span>
              <span />
            </div>
            {keys.map((row) => {
              const revoked = Boolean(row.revoked_at);
              const expired = row.expires_at && new Date(row.expires_at) < new Date();
              const state = revoked ? "revoked" : expired ? "expired" : "active";
              const boundModel = models.find((m) => m.id === row.model_id);
              return (
                <div className="key-row" key={row.id}>
                  <span className="key-name">{row.name}</span>
                  <span className="mono">{row.prefix}…</span>
                  <span className="key-scopes">
                    {(row.scopes || []).map((s) => (
                      <span className="chip" key={s}>
                        {s}
                      </span>
                    ))}
                  </span>
                  <span className="mono">{boundModel ? boundModel.name : "any"}</span>
                  <span className="mono">{row.request_count ?? 0}</span>
                  <span className="mono key-muted">{fmtDate(row.last_used_at)}</span>
                  <span className={`chip ${state === "active" ? "ready" : "failed"}`}>{state}</span>
                  <span className="key-actions">
                    {!revoked && (
                      <>
                        <button className="link-btn" onClick={() => onRotate(row)}>
                          rotate
                        </button>
                        <button className="link-btn danger" onClick={() => onRevoke(row)}>
                          revoke
                        </button>
                      </>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="section">
        <h2 className="section-title">Audit log</h2>
        {audit.length === 0 ? (
          <div className="empty-state">No security events yet.</div>
        ) : (
          <div className="audit-list">
            {audit.map((row) => (
              <div className="audit-row" key={row.id}>
                <span className={`chip ${row.status === "SUCCESS" ? "ready" : "failed"}`}>
                  {row.status.toLowerCase()}
                </span>
                <span className="mono">{row.action}</span>
                <span className="audit-detail">{row.detail || row.resource || ""}</span>
                <span className="log-time">{fmtDate(row.timestamp)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {created && (
        <KeyModal
          title={`Key created: ${created.name}`}
          value={created.key}
          onClose={() => setCreated(null)}
        />
      )}
    </div>
  );
}
