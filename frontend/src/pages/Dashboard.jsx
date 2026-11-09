import { useCallback, useEffect, useRef, useState } from "react";
import { listFormats, listModels, uploadModel } from "../api/client.js";
import StatusChip from "../components/StatusChip.jsx";

function formatBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Dashboard({ onOpenModel }) {
  const [models, setModels] = useState(null);
  const [formats, setFormats] = useState([]);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [query, setQuery] = useState("");
  const [frameworkFilter, setFrameworkFilter] = useState("");
  const fileInputRef = useRef(null);

  const extensions = formats.flatMap((f) => f.extensions || []);
  const accept = extensions.length ? extensions.join(",") : ".pkl,.joblib,.onnx";
  const formatHint = formats.length
    ? formats.map((f) => f.extensions.join(" ")).join(" · ")
    : ".pkl · .joblib · .onnx";

  const refresh = useCallback(async () => {
    try {
      const data = await listModels({ q: query, framework: frameworkFilter });
      setModels(data);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [query, frameworkFilter]);

  useEffect(() => {
    listFormats()
      .then((data) => setFormats(data.formats || []))
      .catch(() => setFormats([]));
  }, []);

  useEffect(() => {
    const timer = setTimeout(refresh, 200);
    return () => clearTimeout(timer);
  }, [refresh]);

  async function handleFile(file) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const name = file.name.replace(/\.[^.]+$/, "");
      await uploadModel(file, name);
      await refresh();
    } catch (e) {
      setError(
        e.status === 401
          ? "Upload requires an API key with the models:write scope. Unlock it on the API keys page."
          : e.message
      );
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Model registry</h1>
        <p className="page-desc">
          Upload a trained model artifact. Each upload is detected, sandbox-validated, stored with a
          SHA-256 checksum, and given an inference endpoint automatically. Uploads require an API
          key with the <code>models:write</code> scope.
        </p>
      </div>

      <div
        className={`upload-panel ${dragOver ? "dragover" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="upload-copy">
          <span className="upload-title">Drop a model file, or choose one</span>
          <span className="upload-hint">{formatHint}</span>
        </div>
        <button className="btn" disabled={uploading} onClick={() => fileInputRef.current?.click()}>
          {uploading ? "Analyzing…" : "Upload model"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="file-input"
          accept={accept}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {error && <p className="error-banner">{error}</p>}

      <div className="registry-toolbar">
        <input
          className="registry-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name or slug…"
        />
        <select
          className="registry-filter"
          value={frameworkFilter}
          onChange={(e) => setFrameworkFilter(e.target.value)}
        >
          <option value="">all frameworks</option>
          {formats.map((f) => (
            <option key={f.framework} value={f.framework}>
              {f.framework}
            </option>
          ))}
        </select>
        <span className="key-muted">
          {models === null ? "" : `${models.length} model${models.length === 1 ? "" : "s"}`}
        </span>
      </div>

      <div className="ledger">
        <div className="ledger-row header">
          <span>Model</span>
          <span>Framework</span>
          <span>Size</span>
          <span>Status</span>
        </div>

        {models === null && <div className="empty-state">Loading registry…</div>}

        {models && models.length === 0 && (
          <div className="empty-state">No models minted yet. Upload one above to get started.</div>
        )}

        {models?.map((m) => (
          <div className="ledger-row" key={m.id} onClick={() => onOpenModel(m.slug || m.id)}>
            <div className="ledger-name">
              {m.name}
              <span className="slug">/api/models/{m.slug}/predict</span>
            </div>
            <span className="ledger-meta">{m.framework}</span>
            <span className="ledger-meta">{formatBytes(m.file_size_bytes)}</span>
            <StatusChip status={m.status} />
          </div>
        ))}
      </div>
    </div>
  );
}
