// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-container">
      {/* HERO (blur wash only in this section) */}
      <section className="rb-section rb-heroSection">
        <div className="rb-sectionInner">
          <div className="rb-heroContent">
            <div>
              <div className="rb-kicker">
                <span className="rb-kickerDot" />
                Keyword scoring + mapping for agencies
              </div>

              <h1 className="rb-title">RANKEDBOX</h1>

              <p className="rb-heroSub">
                Score keywords fast and move from “big list” to clear priorities.
                Start with the free scorer. Upgrade when you’re ready for mapping.
              </p>

              <div className="rb-ctaRow">
                <Link className="rb-ctaPrimary" href="/try">
                  Try the free scorer <span aria-hidden="true">→</span>
                </Link>

                <Link className="rb-ctaSecondaryDark" href="/pricing">
                  View pricing
                </Link>

                <Link className="rb-ctaSecondaryDark" href="/login">
                  Sign in
                </Link>
              </div>

              <div className="rb-chipRow" aria-label="Highlights">
                <span className="rb-chip">Free tool</span>
                <span className="rb-chip">Agency-friendly workflow</span>
                <span className="rb-chip">Score → Map → Export</span>
              </div>
            </div>

            {/* light “product preview” (glass) */}
            <div className="rb-preview" aria-label="Product preview">
              <div className="rb-previewTitle">Quick preview</div>

              <div className="rb-previewRow">
                <div>
                  <div className="rb-previewLabel">Search Volume (A)</div>
                  <input className="rb-previewInput" defaultValue="50" />
                </div>
                <div>
                  <div className="rb-previewLabel">Keyword Difficulty (B)</div>
                  <input className="rb-previewInput" defaultValue="21" />
                </div>
              </div>

              <div className="rb-previewBar">Score 6 — Tier: Elite</div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION: Why (white) */}
      <section className="rb-section rb-section--white">
        <div className="rb-sectionInner">
          <div className="rb-sectionTop">
            <h2 className="rb-h2">Why RankedBox</h2>
            <div className="rb-divider" />
          </div>

          {/* Bento tiles (light, less “boxy”) */}
          <div className="rb-bento">
            <div className="rb-tile rb-tile--wide">
              <div className="rb-tileTitle">Turn keyword lists into a plan</div>
              <p className="rb-tileText">
                Stop arguing about “what to target.” Get a score and a tier that your team can align on.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">Prioritize faster</div>
              <p className="rb-tileText">
                Quickly filter out noise and focus on opportunities that match your strategy.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">Cleaner handoff</div>
              <p className="rb-tileText">
                Outputs are easy to share with content planning and execution—no extra cleanup.
              </p>
            </div>

            <div className="rb-tile rb-tile--wide">
              <div className="rb-tileTitle">Built for repeatability</div>
              <p className="rb-tileText">
                A workflow your team can run weekly so delivery stays consistent across clients.
              </p>
            </div>

            <div className="rb-tile rb-tile--full">
              <div className="rb-tileTitle">Start free, upgrade when you’re ready</div>
              <p className="rb-tileText">
                Use the public scorer now. Sign in for the full mapping experience when you want to move beyond scoring.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION: How it works (gray) */}
      <section className="rb-section rb-section--gray">
        <div className="rb-sectionInner">
          <div className="rb-sectionTop">
            <h2 className="rb-h2">How it works</h2>
            <div className="rb-divider" />
          </div>

          <div className="rb-bento">
            <div className="rb-tile">
              <div className="rb-tileTitle">1) Score</div>
              <p className="rb-tileText">
                Enter Search Volume + Keyword Difficulty and get a clean tier.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">2) Map</div>
              <p className="rb-tileText">
                Map keywords to your URL set using a structured workflow.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">3) Export</div>
              <p className="rb-tileText">
                Download outputs your team can use immediately for planning and execution.
              </p>
            </div>

            <div className="rb-tile rb-tile--full">
              <div className="rb-tileTitle">Try it now</div>
              <p className="rb-tileText">
                Use the free scorer, then sign in when you’re ready to do more.
              </p>

              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 12 }}>
                <Link className="rb-btn" href="/try">
                  Try the free scorer
                </Link>

                <Link className="rb-btnSecondary" href="/pricing">
                  See pricing
                </Link>

                <Link className="rb-btnSecondary" href="/login">
                  Sign in
                </Link>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 26, color: "var(--muted)", fontSize: 13 }}>
            © {new Date().getFullYear()} RankedBox
          </div>
        </div>
      </section>
    </main>
  );
}