import { useEffect, useState } from "react";
import { listModels, getMetrics, getTimeseries, getLogs } from "../api/client.js";
import Sparkline from "../components/Sparkline.jsx";

function fmt(n, digits = 2) {
  return n == null ? "—" : n.toFixed(digits);
}

export default function Monitoring({ initialModelId }) {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(initialModelId || "");
  const [metrics, setMetrics] = useState(null);
  const [series, setSeries] = useState(null);
  const [logs, setLogs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    listModels().then(setModels).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (initialModelId) setSelected(initialModelId);
  }, [initialModelId]);

  useEffect(() => {
    const modelId = selected || undefined;
    Promise.all([getMetrics(modelId), getTimeseries(modelId), getLogs(modelId)])
      .then(([m, s, l]) => {
        setMetrics(m);
        setSeries(s);
        setLogs(l);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [selected]);

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Monitoring</h1>
        <p className="page-desc">Aggregate metrics computed live from inference logs.</p>
      </div>

      <div style={{ marginBottom: 24 }}>
        <select className="select-inline" value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">All models</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      {error && <p className="error-banner">{error}</p>}

      {metrics && (
        <div className="stat-row">
          <div className="stat">
            <div className="stat-label">Requests</div>
            <div className="stat-value">{metrics.total_requests}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Success</div>
            <div className="stat-value success">{metrics.success_count}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Failed</div>
            <div className="stat-value fail">{metrics.failed_count}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Avg latency</div>
            <div className="stat-value">{fmt(metrics.avg_latency_ms)}<span style={{fontSize: 13}}> ms</span></div>
          </div>
          <div className="stat">
            <div className="stat-label">P95 latency</div>
            <div className="stat-value">{fmt(metrics.p95_latency_ms)}<span style={{fontSize: 13}}> ms</span></div>
          </div>
        </div>
      )}

      <div className="section">
        <h2 className="section-title">Latency, recent requests</h2>
        <div className="sparkline-wrap">
          <Sparkline points={series} />
        </div>
      </div>

      <div className="section">
        <h2 className="section-title">Recent inference log</h2>
        {logs && logs.length === 0 && <div className="empty-state">No requests logged yet.</div>}
        {logs?.map((l) => (
          <div className="log-row" key={l.id}>
            <span className={`chip ${l.status === "SUCCESS" ? "ready" : "failed"}`}>{l.status.toLowerCase()}</span>
            <span className="log-error">{l.error_message || ""}</span>
            <span className="mono">{l.latency_ms.toFixed(2)} ms</span>
            <span className="log-time">{new Date(l.timestamp).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
