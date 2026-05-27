// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-home">
      {/* HERO */}
      <section className="rb-hero">
        <div className="rb-heroInner">
          {/* Left: Copy */}
          <div className="rb-heroCopy">
            <div className="rb-heroKicker">
              <span className="rb-heroKickerDot" />
              <span>Keyword scoring + mapping for agencies</span>
            </div>

            <h1 className="rb-heroTitle">RANKEDBOX</h1>

            <div className="rb-heroRule" />

            <p className="rb-heroSub">
              RankedBox helps agencies move from raw keyword data to clear decisions.
              It gives teams a faster way to score opportunities, prioritize what matters,
              and connect search strategy to the way visibility is evolving across AI-driven discovery.
            </p>

            <div className="rb-heroCtas">
              <Link href="/try" className="rb-btnPrimary">
                Try the free scorer
              </Link>
              <Link href="/pricing" className="rb-btnSecondary">
                View pricing
              </Link>
              <Link href="/login" className="rb-btnSecondary">
                Sign in
              </Link>
            </div>

            <div className="rb-heroMeta">
              Keyword scoring | Tiered prioritization | Prompt visibility | Mentions and citations
            </div>
          </div>

          {/* Right: Visual UI mock (non-interactive, for visual selling) */}
          <div className="rb-heroVisual" aria-label="Product preview visual">
            <div className="rb-uiShell">
              <div className="rb-uiTopbar">
                <div className="rb-uiDots" aria-hidden="true">
                  <span className="rb-uiDot red" />
                  <span className="rb-uiDot yellow" />
                  <span className="rb-uiDot" />
                </div>
                <div className="rb-uiTitle">Strategy Report Preview</div>
              </div>

              <div className="rb-uiBody">
                <div className="rb-uiMetricRow">
                  <div className="rb-uiMetric">
                    <div className="rb-uiMetricLabel">Scored opportunities</div>
                    <div className="rb-uiMetricValue">128</div>
                  </div>

                  <div className="rb-uiMetric">
                    <div className="rb-uiMetricLabel">Mapped pages</div>
                    <div className="rb-uiMetricValue">10</div>
                  </div>
                </div>

                <div className="rb-uiTable">
                  <div className="rb-uiTableHeader">
                    <div>Keyword</div>
                    <div>Vol</div>
                    <div>KD</div>
                    <div>Tier</div>
                  </div>

                  <div className="rb-uiTableRow">
                    <div>job openings baltimore</div>
                    <div>30</div>
                    <div>3</div>
                    <div><span className="rb-badge elite">Elite</span></div>
                  </div>

                  <div className="rb-uiTableRow">
                    <div>how do i become a pca in minnesota</div>
                    <div>70</div>
                    <div>9</div>
                    <div><span className="rb-badge elite">Elite</span></div>
                  </div>

                  <div className="rb-uiTableRow">
                    <div>what is people&apos;s first language</div>
                    <div>70</div>
                    <div>35</div>
                    <div><span className="rb-badge fair">Fair</span></div>
                  </div>
                </div>

                <div className="rb-uiBar">
                  Export-ready report with tiers, strategy, and mapped targets
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 1 (WHITE): Why teams use RankedBox */}
      <section className="rb-sectionWhite">
        <div className="rb-band">
          <div className="rb-sectionHeader">
            <div className="rb-kicker">Why teams use RankedBox</div>
            <h2 className="rb-h2">Built to turn research into a plan your team can actually use</h2>
            <p className="rb-p">
              RankedBox helps agencies move from raw keyword data to clear decisions.
              It gives teams a faster way to score opportunities, prioritize what matters,
              and connect search strategy to the way visibility is evolving across AI-driven discovery.
            </p>
          </div>

          <div className="rb-3col">
            <div>
              <h3>Prioritize faster</h3>
              <p>
                Turn long keyword lists into clear tiers so your team knows what deserves attention first
                and what can wait.
              </p>
            </div>

            <div>
              <h3>Plan for modern search</h3>
              <p>
                Go beyond classic keyword metrics by bringing prompt visibility, mentions, and citations
                into the same planning workflow.
              </p>
            </div>

            <div>
              <h3>Create cleaner handoffs</h3>
              <p>
                Export outputs your team can move directly into content planning, page mapping,
                and execution without extra cleanup.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2 (SOFT): What RankedBox helps you do */}
      <section className="rb-sectionSoft">
        <div className="rb-band tight">
          <div className="rb-sectionHeader">
            <div className="rb-kicker">What RankedBox helps you do</div>
            <h2 className="rb-h2">One workflow for keyword decisions and AI discovery visibility</h2>
          </div>

          <div className="rb-2colLines">
            <div className="rb-lineItem">
              <h3>Score and tier opportunities</h3>
              <p>
                Quickly evaluate search volume and keyword difficulty so teams can separate high-value
                targets from noise.
              </p>
            </div>

            <div className="rb-lineItem">
              <h3>Align teams around what to target</h3>
              <p>
                Give strategists, writers, and stakeholders a clearer view of what matters most so
                execution stays focused.
              </p>
            </div>

            <div className="rb-lineItem">
              <h3>Evaluate prompt visibility, mentions, and citations</h3>
              <p>
                Build strategy around how discovery is changing by looking at the signals that influence
                how brands appear in AI-assisted search experiences.
              </p>
            </div>

            <div className="rb-lineItem">
              <h3>Move from analysis to action</h3>
              <p>
                Export cleaner outputs and use them to support prioritization, mapping decisions, and
                a more repeatable workflow across client accounts.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* FINAL CTA (WHITE) */}
      <section className="rb-sectionWhite">
        <div className="rb-band">
          <h2 className="rb-finalTitle">Start with the free scorer. Build from there.</h2>
          <p className="rb-finalCopy">
            Use the public scorer to evaluate opportunities fast. When you are ready for a deeper workflow,
            sign in and move into the full RankedBox experience.
          </p>

          <div className="rb-finalActions">
            <Link href="/try" className="rb-btnLight">
              Try the free scorer
            </Link>
            <Link href="/pricing" className="rb-btnLight">
              View pricing
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}