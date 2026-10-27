const LABELS = {
  READY: "ready",
  READY_UNVALIDATED: "ready · unvalidated",
  FAILED: "failed",
  ANALYZING: "analyzing",
  UPLOADING: "uploading",
};

const CLASS_MAP = {
  READY: "ready",
  READY_UNVALIDATED: "ready-unvalidated",
  FAILED: "failed",
  ANALYZING: "analyzing",
  UPLOADING: "uploading",
};

export default function StatusChip({ status }) {
  const cls = CLASS_MAP[status] || "analyzing";
  const label = LABELS[status] || status?.toLowerCase();
  return <span className={`chip ${cls}`}>{label}</span>;
}
