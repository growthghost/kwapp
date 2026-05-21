// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-container">
      {/* HERO */}
      <div style={{ maxWidth: 920 }}>
        <h1 className="rb-title">RANKEDBOX</h1>

        <p className="rb-subtitle" style={{ fontSize: 18, marginTop: 12 }}>
          Score keywords fast, then map them to the right pages. Built for agencies who
          want speed, clarity, and repeatable decisions.
        </p>

        <div style={{ display: "flex", gap: 12, marginTop: 18, flexWrap: "wrap" }}>
          <Link href="/try" className="rb-btn" style={{ display: "inline-block" }}>
            Try the free score tool
          </Link>

          <Link
            href="/pricing"
            style={{
              display: "inline-block",
              borderRadius: 12,
              padding: "12px 16px",
              fontWeight: 800,
              border: "1px solid rgba(0,0,0,0.16)",
              color: "var(--ink)",
              background: "transparent",
              textDecoration: "none",
            }}
          >
            View pricing
          </Link>
        </div>
      </div>

      {/* BENEFITS */}
      <div style={{ marginTop: 26, maxWidth: 980 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 14,
          }}
        >
          <div className="rb-card">
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
              Stop guessing
            </div>
            <div style={{ color: "var(--muted)", lineHeight: 1.55 }}>
              Turn messy keyword lists into a clear score + tier so you can decide what’s
              worth targeting.
            </div>
          </div>

          <div className="rb-card">
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
              Faster workflows
            </div>
            <div style={{ color: "var(--muted)", lineHeight: 1.55 }}>
              Build a repeatable process your team can run every week without reinventing
              the wheel.
            </div>
          </div>

          <div className="rb-card">
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
              Built for agencies
            </div>
            <div style={{ color: "var(--muted)", lineHeight: 1.55 }}>
              Designed for real client work: scoring now, mapping next, and API-backed
              “secret sauce” later.
            </div>
          </div>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="rb-card" style={{ marginTop: 18, maxWidth: 980 }}>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>
          How it works
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={{ color: "var(--muted)", lineHeight: 1.6 }}>
            <div style={{ fontWeight: 700, color: "var(--ink)" }}>1) Score</div>
            Enter Search Volume + Keyword Difficulty and get a clean tier.
          </div>

          <div style={{ color: "var(--muted)", lineHeight: 1.6 }}>
            <div style={{ fontWeight: 700, color: "var(--ink)" }}>2) Map</div>
            Once logged in, we’ll map keywords to your URL set (CSV workflow).
          </div>

          <div style={{ color: "var(--muted)", lineHeight: 1.6 }}>
            <div style={{ fontWeight: 700, color: "var(--ink)" }}>3) Export</div>
            Download outputs your team can ship into content planning immediately.
          </div>

          <div style={{ color: "var(--muted)", lineHeight: 1.6 }}>
            <div style={{ fontWeight: 700, color: "var(--ink)" }}>4) Scale</div>
            The heavy logic stays server-side (API) so your product stays protected.
          </div>
        </div>

        <div style={{ marginTop: 16, display: "flex", gap: 12, flexWrap: "wrap" }}>
          <Link href="/try" className="rb-btn" style={{ display: "inline-block" }}>
            Try it now
          </Link>
          <Link
            href="/login"
            style={{
              display: "inline-block",
              borderRadius: 12,
              padding: "12px 16px",
              fontWeight: 800,
              border: "1px solid rgba(0,0,0,0.16)",
              color: "var(--ink)",
              background: "transparent",
              textDecoration: "none",
            }}
          >
            Sign in
          </Link>
        </div>
      </div>

      {/* FOOTER */}
      <div style={{ marginTop: 26, color: "var(--muted)", fontSize: 13 }}>
        © {new Date().getFullYear()} RankedBox
      </div>

      {/* Simple responsive tweak (keeps it self-contained without editing globals.css) */}
      <style>{`
        @media (max-width: 900px) {
          .rb-container > div:nth-of-type(2) > div {
            grid-template-columns: 1fr !important;
          }
          .rb-card[style*="grid-template-columns: 1fr 1fr"] {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}