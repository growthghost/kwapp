// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-container">
      {/* HERO (colorful blurred background) */}
      <section className="rb-homeBg">
        <div className="rb-homeInner">
          <h1 className="rb-heroTitle">RankedBox</h1>

          <p className="rb-heroSub">
            Score keywords quickly, then map them to the right pages. Built for agencies who want
            cleaner decisions and faster execution.
          </p>

          <div className="rb-ctaRow">
            <Link className="rb-ctaPrimary" href="/try">
              Try the free scorer <span aria-hidden="true">→</span>
            </Link>

            <Link className="rb-ctaSecondary" href="/pricing">
              View pricing
            </Link>

            <Link className="rb-ctaSecondary" href="/login">
              Sign in
            </Link>
          </div>

          <div className="rb-chipRow" aria-label="Highlights">
            <span className="rb-chip">Free scorer</span>
            <span className="rb-chip">CSV-ready workflow</span>
            <span className="rb-chip">Score → Map → Export</span>
          </div>
        </div>
      </section>

      {/* BENEFITS (light, subtle cards) */}
      <div style={{ marginTop: 24, maxWidth: 980 }}>
        <div className="rb-softGrid">
          <div className="rb-softCard">
            <div className="rb-softTitle">Stop guessing</div>
            <div className="rb-softText">
              Turn keyword lists into a clear score and tier so your team knows what to prioritize.
            </div>
          </div>

          <div className="rb-softCard">
            <div className="rb-softTitle">Move faster</div>
            <div className="rb-softText">
              Repeatable workflow your team can run weekly—without rebuilding spreadsheets from scratch.
            </div>
          </div>

          <div className="rb-softCard">
            <div className="rb-softTitle">Built for agencies</div>
            <div className="rb-softText">
              Outputs you can hand to content planning immediately—clean, consistent, and easy to ship.
            </div>
          </div>
        </div>
      </div>

      {/* CTA STRIP */}
      <div className="rb-card" style={{ marginTop: 18, maxWidth: 980 }}>
        <div className="rb-sectionTitle" style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>
          Ready to try it?
        </div>

        <div style={{ color: "var(--muted)", lineHeight: 1.55 }}>
          Use the free scorer now. Sign in when you’re ready to use the full mapping workflow.
        </div>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 14 }}>
          <Link className="rb-btn" href="/try">
            Try the free scorer
          </Link>

          <Link className="rb-btnSecondary" href="/pricing">
            See pricing
          </Link>
        </div>
      </div>

      <div style={{ marginTop: 26, color: "var(--muted)", fontSize: 13 }}>
        © {new Date().getFullYear()} RankedBox
      </div>
    </main>
  );
}