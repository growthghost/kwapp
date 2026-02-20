// web/src/app/components/HeaderNav.js
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

function getCookie(name) {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(new RegExp("(^|; )" + name + "=([^;]*)"));
  return match ? decodeURIComponent(match[2]) : null;
}

export default function HeaderNav() {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [isAuthed, setIsAuthed] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Re-check cookie whenever route changes (login/logout navigation updates the UI)
  useEffect(() => {
    const v = getCookie("rb_session");
    setIsAuthed(v === "1");
  }, [pathname]);

  // Avoid hydration mismatch by not rendering auth-dependent nav until mounted
  if (!mounted) return null;

  return (
    <nav className="rb-nav">
      <Link className={`rb-navlink ${pathname === "/try" ? "is-active" : ""}`} href="/try">
        Try
      </Link>

      <Link
        className={`rb-navlink ${pathname?.startsWith("/pricing") ? "is-active" : ""}`}
        href="/pricing"
      >
        Pricing
      </Link>

      {isAuthed ? (
        <>
          <Link
            className={`rb-navlink ${pathname?.startsWith("/app") ? "is-active" : ""}`}
            href="/app"
          >
            App
          </Link>
          <Link className={`rb-navlink ${pathname === "/logout" ? "is-active" : ""}`} href="/logout">
            Logout
          </Link>
        </>
      ) : (
        <Link className={`rb-navlink ${pathname?.startsWith("/login") ? "is-active" : ""}`} href="/login">
          Login
        </Link>
      )}
    </nav>
  );
}