// web/src/app/try/page.js
"use client";

import { useMemo, useState } from "react";

const LABEL_MAP = {
  6: "Elite",
  5: "Excellent",
  4: "Good",
  3: "Fair",
  2: "Low",
  1: "Very Low",
  0: "Not rated",
};

const COLOR_MAP = {
  6: "#2ecc71",
  5: "#a3e635",
  4: "#facc15",
  3: "#fb923c",
  2: "#f87171",
  1: "#ef4444",
  0: "#9ca3af",
};

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function calcScore(volume, kd) {
  const v = Number.isFinite(volume) ? volume : 0;
  const k = Number.isFinite(kd) ? kd : 100;

  let s = 0;

  if (v >= 10) s += 1;
  if (v >= 50) s += 1;
  if (v >= 200) s += 1;

  if (k <= 60) s += 1;
  if (k <= 40) s += 1;
  if (k <= 25) s += 1;

  return clamp(s, 0, 6);
}

export default function TryPage() {
  const [volume, setVolume] = useState(50);
  const [kd, setKd] = useState(21);
  const [score, setScore] = useState(null);

  const tier = useMemo(() => {
    if (score === null) return null;
    return LABEL_MAP[score] ?? "Not rated";
  }, [score]);

  const barColor = useMemo(() => {
    if (score === null) return "#9ca3af";
    return COLOR_MAP[score] ?? "#9ca3af";
  }, [score]);

  return (
    <main className="rb-container">
      <h1 className="rb-title">Try RankedBox</h1>

      <p className="rb-subtitle">
        This is the public <strong>/try</strong> page. We’ll put the single keyword
        score tool here next.
      </p>

      <div className="rb-card" style={{ marginTop: 18, maxWidth: 820 }}>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 14 }}>
          Single Keyword Score
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
          }}
        >
          <div>
            <label className="rb-label">Search Volume (A)</label>
            <input
              className="rb-input"
              type="number"
              min="0"
              value={volume}
              onChange={(e) => setVolume(parseInt(e.target.value || "0", 10))}
            />
          </div>

          <div>
            <label className="rb-label">Keyword Difficulty (B)</label>
            <input
              className="rb-input"
              type="number"
              min="0"
              value={kd}
              onChange={(e) => setKd(parseInt(e.target.value || "0", 10))}
            />
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <button
            className="rb-btn"
            onClick={() => setScore(calcScore(Number(volume), Number(kd)))}
          >
            Calculate Score
          </button>
        </div>

        {score !== null && (
          <div
            style={{
              marginTop: 16,
              borderRadius: 14,
              padding: 16,
              textAlign: "center",
              fontWeight: 700,
              background: barColor,
              color: "#0B0B0B",
              border: "1px solid rgba(0,0,0,0.10)",
            }}
          >
            Score {score} — Tier: {tier}
          </div>
        )}
      </div>
    </main>
  );
}
