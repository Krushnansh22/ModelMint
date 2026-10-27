/** Small dependency-free SVG latency sparkline -- no charting library
 * needed for a single latency-over-time line. */
export default function Sparkline({ points, height = 120 }) {
  if (!points || points.length < 2) {
    return <div className="loading">Not enough data yet — send a few predictions to see the trend.</div>;
  }

  const width = 800;
  const pad = 8;
  const values = points.map((p) => p.latency_ms);
  const max = Math.max(...values, 1);
  const min = 0;
  const stepX = (width - pad * 2) / (points.length - 1);

  const coords = points.map((p, i) => {
    const x = pad + i * stepX;
    const y = height - pad - ((p.latency_ms - min) / (max - min || 1)) * (height - pad * 2);
    return [x, y, p];
  });

  const path = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} preserveAspectRatio="none">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="var(--line)" strokeWidth="1" />
      <path d={path} fill="none" stroke="var(--mint)" strokeWidth="1.75" />
      {coords.map(([x, y, p], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={p.status === "FAILED" ? 2.5 : 1.6}
          fill={p.status === "FAILED" ? "var(--copper)" : "var(--mint)"}
        />
      ))}
    </svg>
  );
}
