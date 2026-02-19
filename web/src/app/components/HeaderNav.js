// web/src/app/components/HeaderNav.js
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function isActive(pathname, href) {
  // exact match for now (clean + predictable)
  return pathname === href;
}

export default function HeaderNav({ isAuthed }) {
  const pathname = usePathname();

  return (
    <nav className="rb-nav" aria-label="Primary navigation">
      <Link className={isActive(pathname, "/try") ? "rb-navlink is-active" : "rb-navlink"} href="/try">
        Try
      </Link>

      {isAuthed && (
        <Link className={isActive(pathname, "/app") ? "rb-navlink is-active" : "rb-navlink"} href="/app">
          App
        </Link>
      )}

      <Link
        className={isActive(pathname, "/pricing") ? "rb-navlink is-active" : "rb-navlink"}
        href="/pricing"
      >
        Pricing
      </Link>

      {!isAuthed ? (
        <Link className={isActive(pathname, "/login") ? "rb-navlink is-active" : "rb-navlink"} href="/login">
          Login
        </Link>
      ) : (
        <Link className={isActive(pathname, "/logout") ? "rb-navlink is-active" : "rb-navlink"} href="/logout">
          Logout
        </Link>
      )}
    </nav>
  );
}