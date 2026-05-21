// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-container">
      {/* HERO */}
      <section className="rb-hero">
        <div className="rb-heroInner">
          <div className="rb-kicker">
            <span className="rb-kickerDot" />
            Agency keyword scoring + mapping
          </div>

          <h1 className="rb-heroTitle">RankedBox</h1>

          <p className="rb-heroSub">
            Score keywords fast, then map them to the right pages.
            Built for agencies who want speed, clarity, and repeatable decisions.
          </p>

          <div className="rb-ctaRow">
            <Link className="rb-btnPrimary" href="/try">
              Try the free score tool <span aria-hidden="true">→</span>
            </Link>

            <Link className="rb-btnSecondary" href="/pricing">
              View pricing
            </Link>

            <Link className="rb-btnSecondary" href="/login">
              Sign in
            </Link>
          </div>

          <div className="rb-chipRow" aria-label="Highlights">
            <span className="rb-chip">
              <span className="rb-chipStripe" /> Score + tier instantly
            </span>
            <span className="rb-chip">
              <span className="rb-chipStripe yellow" /> Built for CSV workflows
            </span>
            <span className="rb-chip">
              <span className="rb-chipStripe" /> Secret sauce stays server-side
            </span>
          </div>
        </div>
      </section>

      {/* BENEFITS */}
      <div className="rb-sectionHead">
        <h2 className="rb-h2">Why agencies use RankedBox</h2>
        <div className="rb-h2Accent" />
      </div>

      <section className="rb-grid3">
        <div className="rb-feature">
          <div className="rb-featureTitle">Stop guessing</div>
          <div className="rb-featureText">
            Turn messy keyword lists into a clear score + tier so you can decide
            what’s worth targeting.
          </div>
        </div>

        <div className="rb-feature yellow">
          <div className="rb-featureTitle">Faster workflows</div>
          <div className="rb-featureText">
            Build a repeatable process your team can run every week without
            reinventing the wheel.
          </div>
        </div>

        <div className="rb-feature">
          <div className="rb-featureTitle">Built for agencies</div>
          <div className="rb-featureText">
            Designed for real client work: scoring now, mapping next, and
            API-backed “secret sauce” later.
          </div>
        </div>
      </section>

      {/* HOW IT WORKS (DARK) */}
      <div className="rb-sectionHead">
        <h2 className="rb-h2">How it works</h2>
        <div className="rb-h2Accent" />
      </div>

      <section className="rb-darkCard">
        <div style={{ opacity: 0.88 }}>
          Start with the free scorer. Sign in for the mapping workflow.
        </div>

        <div className="rb-steps">
          <div className="rb-step">
            <div className="rb-stepNum">1</div>
            <div style={{ fontWeight: 900, marginBottom: 6 }}>Score</div>
            <div style={{ opacity: 0.86, lineHeight: 1.55 }}>
              Enter Search Volume + Keyword Difficulty and get a clean tier.
            </div>
          </div>

          <div className="rb-step">
            <div className="rb-stepNum yellow">2</div>
            <div style={{ fontWeight: 900, marginBottom: 6 }}>Map</div>
            <div style={{ opacity: 0.86, lineHeight: 1.55 }}>
              Once logged in, map keywords to your URL set (CSV workflow).
            </div>
          </div>

          <div className="rb-step">
            <div className="rb-stepNum">3</div>
            <div style={{ fontWeight: 900, marginBottom: 6 }}>Export</div>
            <div style={{ opacity: 0.86, lineHeight: 1.55 }}>
              Download outputs your team can ship into content planning immediately.
            </div>
          </div>

          <div className="rb-step">
            <div className="rb-stepNum yellow">4</div>
            <div style={{ fontWeight: 900, marginBottom: 6 }}>Scale</div>
            <div style={{ opacity: 0.86, lineHeight: 1.55 }}>
              Heavy logic stays server-side (API) so the product stays protected.
            </div>
          </div>
        </div>

        <div className="rb-ctaRow" style={{ marginTop: 16 }}>
          <Link className="rb-btnPrimary" href="/try">
            Try it now <span aria-hidden="true">→</span>
          </Link>
          <Link className="rb-btnSecondary" href="/login">
            Sign in
          </Link>
        </div>
      </section>

      {/* FOOTER */}
      <div style={{ marginTop: 26, color: "var(--muted)", fontSize: 13 }}>
        © {new Date().getFullYear()} RankedBox
      </div>
    </main>
  );
}