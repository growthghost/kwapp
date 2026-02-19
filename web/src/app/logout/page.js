// web/src/app/logout/page.js
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LogoutPage() {
  const router = useRouter();

  useEffect(() => {
    // Clear the session cookie
    document.cookie = "rb_session=; Max-Age=0; Path=/; SameSite=Lax";

    // Redirect to login (Option B)
    router.replace("/login");
  }, [router]);

  return (
    <main className="rb-container">
      <h1 className="rb-title">Logging out…</h1>
      <p className="rb-subtitle">Redirecting you to the login page.</p>
    </main>
  );
}