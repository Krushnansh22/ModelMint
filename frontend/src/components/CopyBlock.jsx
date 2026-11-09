import { useState } from "react";

export default function CopyBlock({ label, value }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="snippet">
      <div className="snippet-head">
        <span className="kv-label">{label}</span>
        <button className="link-btn" onClick={copy}>
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre>{value}</pre>
    </div>
  );
}
