import { useCallback, useEffect, useRef, useState } from "react";
import { listModels, uploadModel } from "../api/client.js";
import StatusChip from "../components/StatusChip.jsx";

function formatBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Dashboard({ onOpenModel }) {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listModels();
      setModels(data);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleFile(file) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const name = file.name.replace(/\.(pkl|joblib|onnx)$/i, "");
      await uploadModel(file, name);
      await refresh();
    } catch (e) {
      setError(e.message);
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
          Upload a trained sklearn (.pkl / .joblib) or ONNX (.onnx) model. Each upload is detected,
          sandbox-validated, and given an inference endpoint automatically.
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
          <span className="upload-hint">.pkl · .joblib · .onnx</span>
        </div>
        <button className="btn" disabled={uploading} onClick={() => fileInputRef.current?.click()}>
          {uploading ? "Analyzing…" : "Upload model"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="file-input"
          accept=".pkl,.joblib,.onnx"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {error && <p className="error-banner">{error}</p>}

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
