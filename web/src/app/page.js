// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-container">
      {/* HERO (blur wash only in this section) */}
      <section className="rb-section rb-heroSection">
        <div className="rb-sectionInner">
          <div className="rb-heroContent">
            <div className="rb-kicker">
              <span className="rb-kickerDot" />
              Keyword scoring + mapping for agencies
            </div>

            <h1 className="rb-title">RANKEDBOX</h1>

            <p className="rb-heroSub">
              Score keywords fast and move from “big list” to clear priorities. Start with the free
              scorer. Upgrade when you’re ready for mapping. Built for modern search—track keywords
              and AI-driven discovery signals like mentions and citations as query behavior evolves.
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

            {/* Non-CTA meta info (not pills, not buttons) */}
            <div className="rb-metaRow" aria-label="What you get">
              <span className="rb-metaItem">
                <span className="rb-metaDot" /> Scoring + tiers
              </span>
              <span className="rb-metaItem">
                <span className="rb-metaDot yellow" /> Mapping workflow
              </span>
              <span className="rb-metaItem">
                <span className="rb-metaDot" /> Mentions + citations
              </span>
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

          {/* Bento tiles */}
          <div className="rb-bento">
            <div className="rb-tile rb-tile--wide">
              <div className="rb-tileTitle">Turn keyword lists into a plan</div>
              <p className="rb-tileText">
                Stop debating what to target. Get a consistent score and tier your team can align on.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">Prioritize faster</div>
              <p className="rb-tileText">
                Filter noise, focus effort, and move from research to execution without delays.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">Built for modern search</div>
              <p className="rb-tileText">
                Beyond rankings, teams are tracking visibility in AI surfaces—mentions and citations included.
              </p>
            </div>

            <div className="rb-tile rb-tile--wide">
              <div className="rb-tileTitle">Repeatable across clients</div>
              <p className="rb-tileText">
                A workflow your team can run weekly so delivery stays consistent and scalable.
              </p>
            </div>

            <div className="rb-tile rb-tile--full">
              <div className="rb-tileTitle">Start free, upgrade when ready</div>
              <p className="rb-tileText">
                Use the free scorer now. Step into the full mapping workflow when you’re ready to ship.
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
                Enter Search Volume + Keyword Difficulty to get a clear tier.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">2) Map</div>
              <p className="rb-tileText">
                Assign keywords to the right pages with a structured mapping workflow.
              </p>
            </div>

            <div className="rb-tile">
              <div className="rb-tileTitle">3) Export</div>
              <p className="rb-tileText">
                Download outputs your team can use immediately in planning and execution.
              </p>
            </div>

            <div className="rb-tile rb-tile--full">
              <div className="rb-tileTitle">Try it now</div>
              <p className="rb-tileText">
                Start with the free scorer. Sign in when you want to go deeper.
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