// web/src/app/page.js
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="rb-home">
      <section className="rb-home-hero">
        <div className="rb-hero-swoop rb-hero-swoop-left" />
        <div className="rb-hero-swoop rb-hero-swoop-right" />

        <div className="rb-home-hero-inner">
          <div className="rb-home-hero-grid">
            {/* LEFT COPY */}
            <div className="rb-home-copy">
              <div className="rb-home-eyebrow">
                <span className="rb-home-eyebrow-dot" />
                <span>SCORE. TIER. MAP.</span>
              </div>

              <h1 className="rb-home-title">
                Turn keywords and AI search queries
                <br />
                into mapped priorities in <span className="rb-home-title-accent">minutes.</span>
              </h1>

              <div className="rb-home-title-rule" />

              <p className="rb-home-subtitle">
                Score keywords and AI search queries into clear tiers,
                prioritize what matters, map the right opportunities to the
                right pages, and boost your content strategy to be found in 
                traditional and AI-assisted search.
              </p>

              <div className="rb-home-cta-row">
                <Link href="/try" className="rb-home-cta rb-home-cta-primary">
                  Try RANKEDBOX
                </Link>

                <Link
                  href="/pricing"
                  className="rb-home-cta rb-home-cta-secondary"
                >
                  View pricing
                </Link>

                <Link
                  href="/login"
                  className="rb-home-cta rb-home-cta-secondary"
                >
                  Sign in
                </Link>
              </div>

              <div className="rb-home-meta">
                <span>Keyword scoring</span>
                <span>Tiered prioritization</span>
                <span>Mapping workflow</span>
                <span>AI prompt planning</span>
              </div>
            </div>

            {/* RIGHT DASHBOARD MOCK */}
            <div className="rb-home-dash-wrap">
              <div className="rb-home-dash-glow" />

              <div className="rb-home-dash">
                <div className="rb-home-dash-top">
                  <div className="rb-home-dash-brand">
                    <span className="rb-home-dash-brand-dot" />
                    <span>RANKEDBOX</span>
                  </div>

                  <div className="rb-home-dash-tabs">
                    <span className="rb-home-dash-tab">Overview</span>
                    <span className="rb-home-dash-tab is-active">Keywords</span>
                    <span className="rb-home-dash-tab">Groups</span>
                    <span className="rb-home-dash-tab">Pages</span>
                    <span className="rb-home-dash-tab">Reports</span>
                  </div>

                  <div className="rb-home-dash-export">Export</div>
                </div>

                <div className="rb-home-dash-body">
                  <div className="rb-home-dash-metrics">
                    <div className="rb-home-dash-metric">
                      <div className="rb-home-dash-metric-label">
                        Keywords scored
                      </div>
                      <div className="rb-home-dash-metric-value">1,582</div>
                      <div className="rb-home-dash-metric-sub">
                        +12% vs last 7 days
                      </div>
                    </div>

                    <div className="rb-home-dash-metric">
                      <div className="rb-home-dash-metric-label">
                        High priority
                      </div>
                      <div className="rb-home-dash-metric-value">312</div>
                      <div className="rb-home-dash-metric-sub">
                        19.7% of total
                      </div>
                    </div>

                    <div className="rb-home-dash-metric">
                      <div className="rb-home-dash-metric-label">
                        Avg. opportunity
                      </div>
                      <div className="rb-home-dash-metric-value">78</div>
                      <div className="rb-home-dash-metric-sub">Out of 100</div>
                    </div>

                    <div className="rb-home-dash-metric">
                      <div className="rb-home-dash-metric-label">Mapped</div>
                      <div className="rb-home-dash-metric-value">86%</div>
                      <div className="rb-home-dash-metric-sub">
                        1,362 keywords
                      </div>
                    </div>
                  </div>

                  <div className="rb-home-dash-table">
                    <div className="rb-home-dash-table-head">
                      <div>Keyword / Query</div>
                      <div>Vol</div>
                      <div>KD</div>
                      <div>Score</div>
                      <div>Tier</div>
                    </div>

                    <div className="rb-home-dash-row">
                      <div className="rb-home-dash-keyword">
                        best project management tools
                      </div>
                      <div className="rb-home-dash-cell">92</div>
                      <div className="rb-home-dash-cell">18</div>
                      <div>
                        <div className="rb-home-scorebar">
                          <div
                            className="rb-home-scorebar-fill"
                            style={{ width: "82%" }}
                          />
                        </div>
                      </div>
                      <div>
                        <span className="rb-tier rb-tier-elite">Elite</span>
                      </div>
                    </div>

                    <div className="rb-home-dash-row">
                      <div className="rb-home-dash-keyword">
                        ai project management features
                      </div>
                      <div className="rb-home-dash-cell">76</div>
                      <div className="rb-home-dash-cell">24</div>
                      <div>
                        <div className="rb-home-scorebar">
                          <div
                            className="rb-home-scorebar-fill"
                            style={{ width: "74%" }}
                          />
                        </div>
                      </div>
                      <div>
                        <span className="rb-tier rb-tier-elite">Elite</span>
                      </div>
                    </div>

                    <div className="rb-home-dash-row">
                      <div className="rb-home-dash-keyword">
                        project management for startups
                      </div>
                      <div className="rb-home-dash-cell">58</div>
                      <div className="rb-home-dash-cell">35</div>
                      <div>
                        <div className="rb-home-scorebar">
                          <div
                            className="rb-home-scorebar-fill"
                            style={{ width: "56%" }}
                          />
                        </div>
                      </div>
                      <div>
                        <span className="rb-tier rb-tier-fair">Fair</span>
                      </div>
                    </div>

                    <div className="rb-home-dash-row">
                      <div className="rb-home-dash-keyword">
                        remote collaboration checklist
                      </div>
                      <div className="rb-home-dash-cell">42</div>
                      <div className="rb-home-dash-cell">48</div>
                      <div>
                        <div className="rb-home-scorebar">
                          <div
                            className="rb-home-scorebar-fill"
                            style={{ width: "42%" }}
                          />
                        </div>
                      </div>
                      <div>
                        <span className="rb-tier rb-tier-low">Low</span>
                      </div>
                    </div>
                  </div>

                  <div className="rb-home-dash-bottom">
                    <div className="rb-home-dash-bottom-title">
                      Export-ready report with mapped targets
                    </div>
                    <div className="rb-home-dash-bottom-tag">Ready</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 1 */}
      <section className="rb-home-band">
        <div className="rb-home-inner">
          <h2 className="rb-home-center-kicker">
            Smarter data. Clearer priorities. Better results.
          </h2>

          <div className="rb-home-feature-grid">
            <div className="rb-home-feature-card">
              <div className="rb-home-feature-top">
                <div className="rb-home-feature-icon rb-home-feature-icon-red">
                  ◎
                </div>
                <div className="rb-home-feature-title">Prioritize faster</div>
              </div>
              <div className="rb-home-feature-copy">
                Turn long keyword lists into clear tiers so your team knows what
                deserves attention first and what can wait.
              </div>
            </div>

            <div className="rb-home-feature-card">
              <div className="rb-home-feature-top">
                <div className="rb-home-feature-icon rb-home-feature-icon-gold">
                  ✦
                </div>
                <div className="rb-home-feature-title">
                  Plan for modern search
                </div>
              </div>
              <div className="rb-home-feature-copy">
                Go beyond classic keyword metrics by bringing prompt visibility,
                mentions, and citations into the same planning workflow.
              </div>
            </div>

            <div className="rb-home-feature-card">
              <div className="rb-home-feature-top">
                <div className="rb-home-feature-icon rb-home-feature-icon-green">
                  ↗
                </div>
                <div className="rb-home-feature-title">
                  Create cleaner handoffs
                </div>
              </div>
              <div className="rb-home-feature-copy">
                Export outputs your team can move directly into content
                planning, page mapping, and execution without extra cleanup.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2 */}
      <section className="rb-home-band-alt">
        <div className="rb-home-inner">
          <div className="rb-home-kicker">WHAT RANKEDBOX HELPS YOU DO</div>

          <h2 className="rb-home-h2">
            One workflow for keyword decisions and AI discovery visibility
          </h2>

          <div className="rb-home-two-col">
            <div>
              <div className="rb-home-line-item">
                <div className="rb-home-line-title">
                  Score and tier opportunities
                </div>
                <div className="rb-home-line-copy">
                  Quickly evaluate search volume and keyword difficulty so teams
                  can separate high-value targets from noise.
                </div>
              </div>

              <div className="rb-home-line-item">
                <div className="rb-home-line-title">
                  Evaluate prompt visibility, mentions, and citations
                </div>
                <div className="rb-home-line-copy">
                  Build strategy around how discovery is changing by looking at
                  the signals that influence how brands appear in AI-assisted
                  search experiences.
                </div>
              </div>
            </div>

            <div>
              <div className="rb-home-line-item">
                <div className="rb-home-line-title">
                  Align teams around what to target
                </div>
                <div className="rb-home-line-copy">
                  Give strategists, writers, and stakeholders a clearer view of
                  what matters most so execution stays focused.
                </div>
              </div>

              <div className="rb-home-line-item">
                <div className="rb-home-line-title">
                  Move from analysis to action
                </div>
                <div className="rb-home-line-copy">
                  Export cleaner outputs and use them to support prioritization,
                  mapping decisions, and a more repeatable workflow across
                  client accounts.
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 3 */}
      <section className="rb-home-cta-band">
        <div className="rb-home-cta-block">
          <h2 className="rb-home-h2">
            Start with the free scorer. Build from there.
          </h2>

          <div className="rb-home-cta-copy">
            Use the public scorer to evaluate opportunities fast. When you are
            ready for a deeper workflow, sign in and move into the full
            RankedBox experience.
          </div>

          <div className="rb-home-cta-row">
            <Link href="/try" className="rb-home-cta rb-home-cta-primary">
              Try the free scorer
            </Link>

            <Link
              href="/pricing"
              className="rb-home-cta rb-home-cta-secondary"
              style={{ color: "#0b0b0b", borderColor: "rgba(0,0,0,0.14)", background: "#ffffff" }}
            >
              View pricing
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}