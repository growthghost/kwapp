// web/src/app/logout/page.js
"use client";

import { useEffect } from "react";

export default function LogoutPage() {
  useEffect(() => {
    // Clear cookie (must match name + Path)
    document.cookie = "rb_session=; Max-Age=0; Path=/; SameSite=Lax";

    // UX choice B: go to /login
    window.location.href = "/login";
  }, []);

  return (
    <main className="rb-container">
      <h1 className="rb-title">Logging out…</h1>
      <p className="rb-subtitle">Redirecting to login.</p>
    </main>
  );
}