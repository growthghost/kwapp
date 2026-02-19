// web/src/app/layout.js
import "./globals.css";
import Link from "next/link";
import { cookies } from "next/headers";

export const metadata = {
  title: "RANKEDBOX",
  description: "Score keywords by Search Volume and Keyword Difficulty",
};

export default function RootLayout({ children }) {
  // Server-side read of cookie so header can switch Login/Logout + App
  const isAuthed = cookies().get("rb_session")?.value === "1";

  return (
    <html lang="en">
      <body>
        <header className="rb-header">
          <div className="rb-header-inner">
            <Link href="/" className="rb-brand" aria-label="Go to home">
              <img
                className="rb-logo"
                src="/RankedBoxLogoIcon.png"
                alt="RankedBox"
                width={34}
                height={34}
                style={{
                  width: 34,
                  height: 34,
                  objectFit: "contain",
                  display: "block",
                }}
              />
              <span className="rb-brand-name">RANKEDBOX</span>
            </Link>

            <nav className="rb-nav">
              <Link href="/try">Try</Link>
              <Link href="/pricing">Pricing</Link>

              {isAuthed ? (
                <>
                  <Link href="/app">App</Link>
                  <Link href="/logout">Logout</Link>
                </>
              ) : (
                <Link href="/login">Login</Link>
              )}
            </nav>
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}