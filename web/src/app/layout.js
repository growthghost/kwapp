import "./globals.css";

export const metadata = {
  title: "RankedBox",
  description: "Score keywords by Search Volume and Keyword Difficulty",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <header className="rb-header">
          <div className="rb-header-inner">
            <div className="rb-brand">
              {/* Placeholder logo box (we’ll replace with your real logo next) */}
              <div
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: 10,
                  background: "#FFFFFF",
                  display: "grid",
                  placeItems: "center",
                  color: "#0B0B0B",
                  fontWeight: 900,
                }}
                title="RankedBox"
              >
                R
              </div>

              <div className="rb-brand-name">RANKEDBOX</div>
            </div>

            <nav className="rb-nav">
              <a href="/">Home</a>
              <a href="/try">Try</a>
              <a href="/login">Login</a>
            </nav>
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}
