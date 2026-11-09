import { useEffect, useRef, useState } from "react";
import {
  deleteModel,
  getDrift,
  getModel,
  listVersions,
  predict,
  predictBatch,
  promoteVersion,
  rollbackVersion,
  setTraffic,
  uploadVersion,
} from "../api/client.js";
import CopyBlock from "../components/CopyBlock.jsx";
import StatusChip from "../components/StatusChip.jsx";

const CONTRACT_HINTS = {
  text: 'Send {"text": "..."} for one input, or {"texts": ["...", "..."]} for many.',
  vision: 'Send {"image_base64": "<base64 image bytes>"}.',
  embedding: 'Send {"text": "..."} or {"features": [...]}.',
  tabular: 'Send {"features": [...]}. Batch mode expects a JSON array of instance objects.',
};

export default function ModelDetail({ modelId, onBack, onViewMonitoring, onDeleted }) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [payloadText, setPayloadText] = useState("");
  const [result, setResult] = useState(null);
  const [resultError, setResultError] = useState(null);
  const [running, setRunning] = useState(false);
  const [batchMode, setBatchMode] = useState(false);
  const [versions, setVersions] = useState([]);
  const [drift, setDrift] = useState(null);
  const [governanceMessage, setGovernanceMessage] = useState(null);
  const [governanceError, setGovernanceError] = useState(null);
  const [weights, setWeights] = useState({});
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setModel(null);
    getModel(modelId)
      .then((m) => {
        if (cancelled) return;
        setModel(m);
        setPayloadText(JSON.stringify(m.example_request ?? { features: [] }, null, 2));
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  useEffect(() => {
    if (!model?.id) return;
    let cancelled = false;
    Promise.all([
      listVersions(model.id).catch(() => null),
      getDrift(model.id).catch(() => null),
    ]).then(([versionData, driftData]) => {
      if (cancelled) return;
      const list = versionData?.versions || [];
      setVersions(list);
      setWeights(Object.fromEntries(list.map((v) => [v.id, v.traffic_weight ?? 0])));
      setDrift(driftData);
    });
    return () => {
      cancelled = true;
    };
  }, [model?.id]);

  async function refreshGovernance() {
    const [versionData, driftData] = await Promise.all([
      listVersions(model.id).catch(() => null),
      getDrift(model.id).catch(() => null),
    ]);
    const list = versionData?.versions || [];
    setVersions(list);
    setWeights(Object.fromEntries(list.map((v) => [v.id, v.traffic_weight ?? 0])));
    setDrift(driftData);
  }

  async function handlePromote(version) {
    setGovernanceMessage(null);
    setGovernanceError(null);
    try {
      await promoteVersion(version.id);
      await refreshGovernance();
      setGovernanceMessage(`Promoted v${version.version}.`);
    } catch (e) {
      setGovernanceError(e.message);
    }
  }

  async function handleRollback(version) {
    setGovernanceMessage(null);
    setGovernanceError(null);
    try {
      const res = await rollbackVersion(version.id);
      await refreshGovernance();
      setGovernanceMessage(`Rolled back to ${res.rolled_back_to ? "the previous version" : "v" + version.version}.`);
    } catch (e) {
      setGovernanceError(e.message);
    }
  }

  async function handleApplyTraffic() {
    setGovernanceMessage(null);
    setGovernanceError(null);
    try {
      const payload = {};
      versions.forEach((v) => {
        payload[v.id] = Number(weights[v.id] ?? 0);
      });
      await setTraffic(model.id, payload);
      await refreshGovernance();
      setGovernanceMessage("Traffic split applied.");
    } catch (e) {
      setGovernanceError(e.message);
    }
  }

  async function handleVersionUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setGovernanceMessage(null);
    setGovernanceError(null);
    try {
      const created = await uploadVersion(model.family_id || model.id, file);
      await refreshGovernance();
      setGovernanceMessage(`Uploaded v${created.version} (${created.status}). Promote it when ready.`);
    } catch (e) {
      setGovernanceError(e.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function runPredict() {
    setRunning(true);
    setResult(null);
    setResultError(null);
    try {
      const parsed = JSON.parse(payloadText);
      const res = batchMode
        ? await predictBatch(modelId, Array.isArray(parsed) ? parsed : [parsed])
        : await predict(modelId, parsed);
      setResult(res);
    } catch (e) {
      setResultError(e.message);
    } finally {
      setRunning(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete "${model.name}" and its stored artifact? This cannot be undone.`)) {
      return;
    }
    try {
      await deleteModel(model.id);
      onDeleted?.();
    } catch (e) {
      setResultError(e.message);
    }
  }

  if (error) return <p className="error-banner">{error}</p>;
  if (!model) return <p className="loading">Loading model…</p>;

  const servable = model.status === "READY" || model.status === "READY_UNVALIDATED";
  const fields = model.input_schema?.fields || [];
  const example = model.example_request ?? { features: [] };
  const baseUrl = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";
  const predictUrl = `${baseUrl}/api/models/${model.slug}/predict`;
  const bodyJson = JSON.stringify(example);
  const curlSnippet =
    `curl -X POST ${predictUrl} \\\n` +
    `  -H "X-API-Key: $MODELMINT_API_KEY" \\\n` +
    `  -H "Content-Type: application/json" \\\n` +
    `  -d '${bodyJson}'`;
  const pythonSnippet =
    `import requests\n\n` +
    `resp = requests.post(\n` +
    `    "${predictUrl}",\n` +
    `    headers={"X-API-Key": "mm_live_..."},\n` +
    `    json=${bodyJson},\n` +
    `)\n` +
    `print(resp.status_code, resp.json())`;

  return (
    <div>
      <button className="back-link" onClick={onBack}>
        ← registry
      </button>

      <div className="detail-head">
        <div>
          <h1 className="detail-title">{model.name}</h1>
          <div className="detail-slug mono">/api/models/{model.slug}/predict</div>
        </div>
        <StatusChip status={model.status} />
      </div>

      {model.failure_reason && <div className="reason-line">{model.failure_reason}</div>}

      <div className="section">
        <h2 className="section-title">Metadata</h2>
        <div className="kv-grid">
          <div className="kv">
            <span className="kv-label">Framework</span>
            <span className="kv-value">{model.framework}</span>
          </div>
          <div className="kv">
            <span className="kv-label">Model type</span>
            <span className="kv-value">{model.model_type || "—"}</span>
          </div>
          <div className="kv">
            <span className="kv-label">Task</span>
            <span className="kv-value">{model.task_type || "—"}</span>
          </div>
          <div className="kv">
            <span className="kv-label">Artifact</span>
            <span className="kv-value">{model.artifact_format || "—"}</span>
          </div>
          <div className="kv">
            <span className="kv-label">Schema confidence</span>
            <span className="kv-value">{model.schema_confidence || "—"}</span>
          </div>
          <div className="kv">
            <span className="kv-label">Load time</span>
            <span className="kv-value">
              {model.load_time_ms != null ? `${model.load_time_ms.toFixed(1)} ms` : "—"}
            </span>
          </div>
          {model.metadata?.n_features != null && (
            <div className="kv">
              <span className="kv-label">Features</span>
              <span className="kv-value">{model.metadata.n_features}</span>
            </div>
          )}
          {model.metadata?.n_classes != null && (
            <div className="kv">
              <span className="kv-label">Classes</span>
              <span className="kv-value">{model.metadata.n_classes}</span>
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h2 className="section-title">Versions &amp; traffic</h2>
          <div className="inline-actions">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden-input"
              onChange={handleVersionUpload}
            />
            <button
              className="btn secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? "Uploading…" : "Upload new version"}
            </button>
          </div>
        </div>

        {governanceMessage && <p className="governance-note">{governanceMessage}</p>}
        {governanceError && <p className="result-error">{governanceError}</p>}

        {versions.length > 0 && (
          <>
            <div className="version-table">
              <div className="version-row header">
                <span>Version</span>
                <span>Status</span>
                <span>Active</span>
                <span>Traffic</span>
                <span />
              </div>
              {versions.map((v) => {
                const canRollback = versions.some(
                  (other) =>
                    other.version < v.version &&
                    ["READY", "READY_UNVALIDATED"].includes(other.status)
                );
                return (
                  <div className="version-row" key={v.id}>
                    <span className="mono">v{v.version}</span>
                    <StatusChip status={v.status} />
                    <span>
                      {v.is_active ? <span className="active-badge">active</span> : "—"}
                    </span>
                    <input
                      className="weight-input"
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      value={weights[v.id] ?? 0}
                      onChange={(e) =>
                        setWeights((prev) => ({ ...prev, [v.id]: e.target.value }))
                      }
                    />
                    <span className="key-actions">
                      <button
                        className="link-btn"
                        onClick={() => handlePromote(v)}
                        disabled={v.is_active}
                      >
                        promote
                      </button>
                      <button
                        className="link-btn"
                        onClick={() => handleRollback(v)}
                        disabled={!canRollback}
                      >
                        rollback
                      </button>
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="traffic-actions">
              <button className="btn secondary" onClick={handleApplyTraffic}>
                Apply traffic split
              </button>
              <span className="key-muted">
                Family alias:{" "}
                <code className="mono">
                  /api/families/{model.family_id || model.id}/predict
                </code>
              </span>
            </div>
          </>
        )}
      </div>

      <div className="section">
        <h2 className="section-title">Input drift</h2>
        {!drift ? (
          <p className="key-muted">Drift analysis is available after requests are served.</p>
        ) : (
          <>
            <span className={`drift-status ${drift.status.toLowerCase()}`}>{drift.status}</span>
            <p className="key-muted drift-meta">
              baseline {drift.baseline_count} · current {drift.current_count} · threshold{" "}
              {drift.threshold}
              {drift.reason ? ` · ${drift.reason}` : ""}
            </p>
            {drift.features?.length > 0 && (
              <div className="version-table drift-table">
                <div className="version-row drift-row header">
                  <span>Feature</span>
                  <span>Baseline</span>
                  <span>Current</span>
                  <span>|z|</span>
                  <span>Status</span>
                </div>
                {drift.features.map((f) => (
                  <div className="version-row drift-row" key={f.feature}>
                    <span className="mono">{f.feature}</span>
                    <span>{f.baseline_mean}</span>
                    <span>{f.current_mean}</span>
                    <span>{f.z_score}</span>
                    <span className={f.drift ? "drift-flag" : "key-muted"}>
                      {f.drift ? "drift" : "ok"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {fields.length > 0 && (
        <div className="section">
          <h2 className="section-title">Input schema</h2>
          <div className="schema-fields">
            {fields.map((f, i) => (
              <div className="schema-field-row" key={i}>
                <span className="name">{f.name || `field ${i}`}</span>
                <span className="dtype">{f.dtype || "?"}</span>
                <span className="shape">{f.shape ? `[${f.shape.map((s) => s ?? "?").join(", ")}]` : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="section">
        <h2 className="section-title">Try it</h2>
        {servable ? (
          <div className="tryit">
            <p className="key-hint">{CONTRACT_HINTS[model.task_type] || CONTRACT_HINTS.tabular}</p>
            <label className="scope-toggle batch-toggle">
              <input
                type="checkbox"
                checked={batchMode}
                onChange={(e) => setBatchMode(e.target.checked)}
              />
              Batch mode (POST …/predict/batch)
            </label>
            <textarea
              value={payloadText}
              onChange={(e) => setPayloadText(e.target.value)}
              spellCheck={false}
            />
            <div className="tryit-actions">
              <button className="btn" onClick={runPredict} disabled={running}>
                {running ? "Running…" : "Send request"}
              </button>
              {result && <span className="result-latency">{result.latency_ms} ms {result.cold_start ? "· cold start" : ""}</span>}
            </div>

            {resultError && (
              <div className="result-block">
                <p className="result-error">{resultError}</p>
              </div>
            )}

            {result && (
              <div className="result-block">
                <span className="kv-label">Prediction</span>
                <pre>{JSON.stringify(result.predictions, null, 2)}</pre>
              </div>
            )}
          </div>
        ) : (
          <p className="empty-state" style={{ border: "none", padding: "12px 0" }}>
            This model isn't servable yet (status: {model.status}).
          </p>
        )}
      </div>

      {servable && (
        <div className="section">
          <h2 className="section-title">Request examples</h2>
          <CopyBlock label="curl" value={curlSnippet} />
          <CopyBlock label="python" value={pythonSnippet} />
        </div>
      )}

      <div className="section detail-actions">
        <button className="btn secondary" onClick={() => onViewMonitoring(model.id)}>
          View monitoring for this model →
        </button>
        <button className="btn danger" onClick={handleDelete}>
          Delete model
        </button>
      </div>
    </div>
  );
}
