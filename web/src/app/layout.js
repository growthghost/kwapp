// web/src/app/layout.js
import "./globals.css";

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
            <div className="rb-brand">
              <img className="rb-logo" src="/RankedBoxLogoIcon.png" alt="RankedBox" />
              <span className="rb-brand-name">RANKEDBOX</span>
            </div>

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
