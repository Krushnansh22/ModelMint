import { useState } from "react";
import Dashboard from "./pages/Dashboard.jsx";
import ModelDetail from "./pages/ModelDetail.jsx";
import Monitoring from "./pages/Monitoring.jsx";
import ApiKeys from "./pages/ApiKeys.jsx";

const NAV = [
  { id: "registry", label: "Registry" },
  { id: "monitoring", label: "Monitoring" },
  { id: "keys", label: "API keys" },
];

export default function App() {
  const [view, setView] = useState("registry"); // registry | detail | monitoring
  const [selectedModel, setSelectedModel] = useState(null);
  const [monitoringModelId, setMonitoringModelId] = useState(null);

  function openModel(idOrSlug) {
    setSelectedModel(idOrSlug);
    setView("detail");
  }

  function openMonitoring(modelId) {
    setMonitoringModelId(modelId || null);
    setView("monitoring");
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div>
          <div className="brand">
            <span className="brand-mark">M</span>
            <span className="brand-name">ModelMint</span>
          </div>
          <div className="brand-sub">model-as-a-service</div>
        </div>

        <nav className="nav">
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${view === item.id || (view === "detail" && item.id === "registry") ? "active" : ""}`}
              onClick={() => {
                if (item.id === "monitoring") openMonitoring(null);
                else if (item.id === "keys") setView("keys");
                else setView("registry");
              }}
            >
              <span className="nav-dot" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          scoped API keys
          <br />
          security build
        </div>
      </aside>

      <main className="main">
        {view === "registry" && <Dashboard onOpenModel={openModel} />}
        {view === "detail" && (
          <ModelDetail
            modelId={selectedModel}
            onBack={() => setView("registry")}
            onViewMonitoring={openMonitoring}
            onDeleted={() => setView("registry")}
          />
        )}
        {view === "monitoring" && <Monitoring initialModelId={monitoringModelId} />}
        {view === "keys" && <ApiKeys />}
      </main>
    </div>
  );
}
