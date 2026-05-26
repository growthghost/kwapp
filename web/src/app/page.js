// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-home">
      {/* HERO */}
      <section className="rb-home-hero">
        <div className="rb-home-hero-inner">
          <div className="rb-home-hero-copy">
            <div className="rb-home-kicker">
              <span className="rb-home-kicker-dot" />
              <span>Keyword & AI Prompt Scoring + Mapping for Agencies</span>
            </div>

            <h1 className="rb-home-title">RANKEDBOX</h1>

            <div className="rb-home-rule" />

            <p className="rb-home-subhead">
              Get clarity fast. Score and sort keywords and AI queries into clear tiers 
              using volume and difficulty, then map priorities to the right pages 
              so your team can execute.
            </p>

            <div className="rb-home-cta-row">
              <Link href="/try" className="rb-home-cta-primary">
                Try RANKEDBOX
              </Link>

              <Link href="/pricing" className="rb-home-cta-secondary">
                View pricing
              </Link>

              <Link href="/login" className="rb-home-cta-secondary">
                Sign in
              </Link>
            </div>

            <div className="rb-home-meta">
              Keyword & Prompt Scoring | Tiered Prioritization | Prompt visibility |
              Mentions and citations
            </div>
          </div>
        </div>
      </section>

      {/* VALUE SECTION */}
      <section className="rb-home-section-white">
        <div className="rb-home-band">
          <div className="rb-home-section-header">
            <div className="rb-home-section-kicker">Why teams use RankedBox</div>
            <h2 className="rb-home-section-title">
              Built to turn research into a plan your team can actually use
            </h2>
            <p className="rb-home-section-copy">
              RankedBox helps agencies move from raw keyword data to clear
              decisions. It gives teams a faster way to score opportunities,
              prioritize what matters, and connect search strategy to the way
              visibility is evolving across AI-driven discovery.
            </p>
          </div>

          <div className="rb-home-value-grid">
            <div className="rb-home-value-item">
              <h3 className="rb-home-value-title">Prioritize faster</h3>
              <p className="rb-home-value-copy">
                Turn long keyword lists into clear tiers so your team knows what
                deserves attention first and what can wait.
              </p>
            </div>

            <div className="rb-home-value-item">
              <h3 className="rb-home-value-title">Plan for modern search</h3>
              <p className="rb-home-value-copy">
                Go beyond classic keyword metrics by bringing prompt visibility,
                mentions, and citations into the same planning workflow.
              </p>
            </div>

            <div className="rb-home-value-item">
              <h3 className="rb-home-value-title">Create cleaner handoffs</h3>
              <p className="rb-home-value-copy">
                Export outputs your team can move directly into content
                planning, page mapping, and execution without extra cleanup.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* DETAIL SECTION */}
      <section className="rb-home-section-soft">
        <div className="rb-home-band rb-home-band-tight">
          <div className="rb-home-section-header">
            <div className="rb-home-section-kicker">What RankedBox helps you do</div>
            <h2 className="rb-home-section-title">
              One workflow for keyword decisions and AI discovery visibility
            </h2>
          </div>

          <div className="rb-home-detail-grid">
            <div className="rb-home-detail-item">
              <h3 className="rb-home-detail-item-title">
                Score and tier opportunities
              </h3>
              <p className="rb-home-detail-item-copy">
                Quickly evaluate search volume and keyword difficulty so teams
                can separate high-value targets from noise.
              </p>
            </div>

            <div className="rb-home-detail-item">
              <h3 className="rb-home-detail-item-title">
                Align teams around what to target
              </h3>
              <p className="rb-home-detail-item-copy">
                Give strategists, writers, and stakeholders a clearer view of
                what matters most so execution stays focused.
              </p>
            </div>

            <div className="rb-home-detail-item">
              <h3 className="rb-home-detail-item-title">
                Evaluate prompt visibility, mentions, and citations
              </h3>
              <p className="rb-home-detail-item-copy">
                Build strategy around how discovery is changing by looking at
                the signals that influence how brands appear in AI-assisted
                search experiences.
              </p>
            </div>

            <div className="rb-home-detail-item">
              <h3 className="rb-home-detail-item-title">
                Move from analysis to action
              </h3>
              <p className="rb-home-detail-item-copy">
                Export cleaner outputs and use them to support prioritization,
                mapping decisions, and a more repeatable workflow across client
                accounts.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CLOSING CTA */}
      <section className="rb-home-cta-band">
        <div className="rb-home-band">
          <div className="rb-home-cta-block">
            <h2 className="rb-home-cta-title">
              Start with the free scorer. Build from there.
            </h2>

            <p className="rb-home-cta-copy">
              Use the public scorer to evaluate opportunities fast. When you are
              ready for a deeper workflow, sign in and move into the full
              RankedBox experience.
            </p>

            <div className="rb-home-cta-actions">
              <Link href="/try" className="rb-home-cta-primary">
                Try the free scorer
              </Link>

              <Link href="/pricing" className="rb-home-cta-secondary" style={{ color: "#0b0b0b", borderColor: "rgba(0,0,0,0.14)", background: "transparent" }}>
                View pricing
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}