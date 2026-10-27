import { useEffect, useState } from "react";
import { getModel, predict } from "../api/client.js";
import StatusChip from "../components/StatusChip.jsx";

export default function ModelDetail({ modelId, onBack, onViewMonitoring }) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [payloadText, setPayloadText] = useState("");
  const [result, setResult] = useState(null);
  const [resultError, setResultError] = useState(null);
  const [running, setRunning] = useState(false);

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

  async function runPredict() {
    setRunning(true);
    setResult(null);
    setResultError(null);
    try {
      const payload = JSON.parse(payloadText);
      const res = await predict(modelId, payload);
      setResult(res);
    } catch (e) {
      setResultError(e.message);
    } finally {
      setRunning(false);
    }
  }

  if (error) return <p className="error-banner">{error}</p>;
  if (!model) return <p className="loading">Loading model…</p>;

  const servable = model.status === "READY" || model.status === "READY_UNVALIDATED";
  const fields = model.input_schema?.fields || [];

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

      <div className="section">
        <button className="btn secondary" onClick={() => onViewMonitoring(model.id)}>
          View monitoring for this model →
        </button>
      </div>
    </div>
  );
}
