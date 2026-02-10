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

// Used for the single-word color card (unchanged, since these represent tiers)
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

/**
 * TEMP scoring logic for /try (front-end only).
 * We'll replace this with a secure API call later.
 *
 * For now: simple 0–6 score based on low KD + decent volume.
 */
function calcScore(volume, kd) {
  const v = Number.isFinite(volume) ? volume : 0;
  const k = Number.isFinite(kd) ? kd : 100;

  // basic heuristics
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
    <div className="page">
      <div className="container">
        <h1 className="h1">Try RankedBox</h1>
        <p className="sub">
          This is the public <strong>/try</strong> page. It uses a temporary
          front-end-only score for now. Later, we’ll call your secure Python API.
        </p>

        <div className="card">
          <div className="sectionTitle">Single Keyword Score</div>

          <div className="grid2">
            <div>
              <label className="label">Search Volume (A)</label>
              <input
                className="input"
                type="number"
                min="0"
                value={volume}
                onChange={(e) => setVolume(parseInt(e.target.value || "0", 10))}
              />
            </div>

            <div>
              <label className="label">Keyword Difficulty (B)</label>
              <input
                className="input"
                type="number"
                min="0"
                value={kd}
                onChange={(e) => setKd(parseInt(e.target.value || "0", 10))}
              />
            </div>
          </div>

          <div className="btnRow">
            <button
              className="btn"
              onClick={() => setScore(calcScore(Number(volume), Number(kd)))}
            >
              Calculate Score
            </button>
          </div>

          {score !== null && (
            <div className="scoreBar" style={{ background: barColor }}>
              Score {score} — Tier: {tier}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
