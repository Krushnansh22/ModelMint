import { useState } from "react";

export default function KeyModal({ title, value, onClose }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2 className="section-title">{title}</h2>
        <p className="modal-warning">
          Copy this key now. It is shown once and cannot be recovered.
        </p>
        <div className="key-reveal">
          <code>{value}</code>
        </div>
        <div className="modal-actions">
          <button className="btn" onClick={copy}>
            {copied ? "Copied" : "Copy key"}
          </button>
          <button className="btn secondary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
