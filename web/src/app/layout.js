// web/src/app/layout.js
import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "RANKEDBOX",
  description: "Score keywords by Search Volume and Keyword Difficulty",
};

export default function RootLayout({ children }) {
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
              <a href="/try">Try</a>
              <a href="/app">App</a>
              <a href="/pricing">Pricing</a>
              <a href="/login">Login</a>
            </nav>
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}
