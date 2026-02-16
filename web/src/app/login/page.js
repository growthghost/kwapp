// web/src/app/login/page.js
"use client";

import { useState } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");

  return (
    <main className="rb-container">
      <h1 className="rb-title">Login</h1>
      <p className="rb-subtitle">
        Placeholder login UI. Later we’ll wire this to auth + your API.
      </p>

      <div className="rb-card" style={{ marginTop: 18, maxWidth: 520 }}>
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <label className="rb-label">Email</label>
            <input
              className="rb-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
            />
          </div>

          <div>
            <label className="rb-label">Password</label>
            <input
              className="rb-input"
              type="password"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          <button
            className="rb-btn"
            onClick={() => alert("Placeholder only — auth coming next.")}
            style={{ marginTop: 6 }}
          >
            Sign in
          </button>

          <div style={{ color: "var(--muted)", fontSize: 13 }}>
            Next: add real auth (Clerk / NextAuth) + protect <code>/app</code>.
          </div>
        </div>
      </div>
    </main>
  );
}
