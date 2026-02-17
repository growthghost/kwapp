// web/src/app/login/page.js
"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");

  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") || "/app";

  function handleSignIn() {
    // TEMP AUTH (Milestone A):
    // Set a simple cookie to simulate "logged in"
    const days = 21;
    const maxAge = days * 24 * 60 * 60; // seconds

    document.cookie = `rb_session=1; Max-Age=${maxAge}; Path=/; SameSite=Lax`;

    router.push(nextPath);
  }

  return (
    <main className="rb-container">
      <h1 className="rb-title">Login</h1>
      <p className="rb-subtitle">
        Placeholder login. This sets a temporary session cookie for {`21`} days.
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
              placeholder="you@agency.com"
              autoComplete="off"
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
              autoComplete="off"
            />
          </div>

          <button className="rb-btn" onClick={handleSignIn} style={{ marginTop: 6 }}>
            Sign in
          </button>

          <div style={{ color: "var(--muted)", fontSize: 13 }}>
            Next: replace this with real auth + database users.
          </div>
        </div>
      </div>
    </main>
  );
}
