// web/src/app/page.js
import Link from "next/link";

function IconStar({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2.8l2.9 6.1 6.7.9-4.9 4.7 1.2 6.7L12 18.9 6.1 21.2l1.2-6.7-4.9-4.7 6.7-.9L12 2.8z"
        fill="currentColor"
        opacity="0.95"
      />
    </svg>
  );
}

function IconGauge({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 4a10 10 0 00-10 10v2h20v-2A10 10 0 0012 4z"
        stroke="currentColor"
        strokeWidth="2"
        opacity="0.9"
      />
      <path d="M12 14l5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="14" r="2" fill="currentColor" />
    </svg>
  );
}

function IconLink({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M10.6 13.4a4 4 0 010-5.7l1.1-1.1a4 4 0 015.7 0"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M13.4 10.6a4 4 0 010 5.7l-1.1 1.1a4 4 0 01-5.7 0"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconExport({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 3v10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M8 7l4-4 4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 14v5h14v-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function IconFilter({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 5h16l-6 7v6l-4 1v-7L4 5z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        opacity="0.95"
      />
    </svg>
  );
}

export default function HomePage() {
  return (
    <main className="rb-home">
      {/* HERO */}
      <section className="rb-hero">
        <div className="rb-heroSwoops" aria-hidden="true" />

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
              RankedBox helps agencies move from raw keyword data to clear decisions. It gives teams a
              faster way to score opportunities, prioritize what matters, and connect search strategy
              to the way visibility is evolving across AI-driven discovery.
            </p>

            <div className="rb-heroCtas">
              <Link href="/try" className="rb-btnPrimary">Try the free scorer</Link>
              <Link href="/pricing" className="rb-btnSecondary">View pricing</Link>
              <Link href="/login" className="rb-btnSecondary">Sign in</Link>
            </div>

            <div className="rb-heroMeta">
              Keyword scoring | Tiered prioritization | Mapping workflow | AI query planning
            </div>
          </div>

          {/* Right: Neon dashboard mock */}
          <div className="rb-heroVisual" aria-label="Product preview visual">
            <div className="rb-uiFrame">
              <div className="rb-uiShell">
                <div className="rb-uiTopbar">
                  <div className="rb-uiBrand">
                    <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                      <span style={{ width: 8, height: 8, background: "linear-gradient(90deg, var(--red), var(--yellow))" }} />
                      <span>RANKEDBOX</span>
                    </span>
                  </div>

                  <div className="rb-uiNav" aria-hidden="true">
                    <span>Overview</span>
                    <span className="active">Keywords</span>
                    <span>Groups</span>
                    <span>Pages</span>
                    <span>Reports</span>
                  </div>

                  <div className="rb-uiActions">
                    <div className="rb-uiBtn"><IconExport /><span>Export Report</span></div>
                    <div className="rb-uiBtn"><IconFilter /><span>Filters</span></div>
                  </div>
                </div>

                <div className="rb-uiBody">
                  <div className="rb-uiCards">
                    <div className="rb-uiCard">
                      <div className="rb-uiCardLabel">
                        <span style={{ color: "rgba(255,255,255,0.82)" }}><IconGauge /></span>
                        <span>Keywords scored</span>
                      </div>
                      <div className="rb-uiCardValue">1,582</div>
                      <div className="rb-uiCardNote">+12% vs last 7 days</div>
                    </div>

                    <div className="rb-uiCard">
                      <div className="rb-uiCardLabel">
                        <span style={{ color: "rgba(255,255,255,0.82)" }}><IconStar /></span>
                        <span>High priority</span>
                      </div>
                      <div className="rb-uiCardValue">312</div>
                      <div className="rb-uiCardNote">19.7% of total</div>
                    </div>

                    <div className="rb-uiCard">
                      <div className="rb-uiCardLabel">
                        <span style={{ color: "rgba(255,255,255,0.82)" }}><IconGauge /></span>
                        <span>Avg. opportunity</span>
                      </div>
                      <div className="rb-uiCardValue">78</div>
                      <div className="rb-uiCardNote">Out of 100</div>
                    </div>

                    <div className="rb-uiCard">
                      <div className="rb-uiCardLabel">
                        <span style={{ color: "rgba(255,255,255,0.82)" }}><IconLink /></span>
                        <span>Mapped</span>
                      </div>
                      <div className="rb-uiCardValue">86%</div>
                      <div className="rb-uiCardNote">1,362 keywords</div>
                    </div>
                  </div>

                  <div className="rb-uiTable">
                    <div className="rb-uiTableHeader">
                      <div>Keyword / Query</div>
                      <div>Vol</div>
                      <div>KD</div>
                      <div>Score</div>
                      <div>Tier</div>
                    </div>

                    <div className="rb-uiTableRow">
                      <div>best project management tools</div>
                      <div>92</div>
                      <div>18</div>
                      <div className="rb-uiScoreBar"><span style={{ width: "78%" }} /></div>
                      <div><span className="rb-tierBadge elite">Elite</span></div>
                    </div>

                    <div className="rb-uiTableRow">
                      <div>ai project management features</div>
                      <div>76</div>
                      <div>24</div>
                      <div className="rb-uiScoreBar"><span style={{ width: "70%" }} /></div>
                      <div><span className="rb-tierBadge elite">Elite</span></div>
                    </div>

                    <div className="rb-uiTableRow">
                      <div>project management for startups</div>
                      <div>58</div>
                      <div>35</div>
                      <div className="rb-uiScoreBar"><span style={{ width: "58%" }} /></div>
                      <div><span className="rb-tierBadge fair">Fair</span></div>
                    </div>

                    <div className="rb-uiTableRow">
                      <div>remote collaboration checklist</div>
                      <div>42</div>
                      <div>48</div>
                      <div className="rb-uiScoreBar"><span style={{ width: "44%" }} /></div>
                      <div><span className="rb-tierBadge low">Low</span></div>
                    </div>
                  </div>

                  <div className="rb-uiFooter">
                    <span>Export-ready report with tiers, strategy, and mapped targets</span>
                    <span style={{ fontSize: 13, fontWeight: 700 }}>CSV • PDF • API</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* SECTION 1 */}
      <section className="rb-sectionWhite">
        <div className="rb-band">
          <div className="rb-sectionHeader">
            <div className="rb-kicker">Why teams use RankedBox</div>
            <h2 className="rb-h2">Built to turn research into a plan your team can actually use</h2>
            <p className="rb-p">
              RankedBox helps agencies move from raw keyword data to clear decisions. It gives teams a faster
              way to score opportunities, prioritize what matters, and connect search strategy to the way
              visibility is evolving across AI-driven discovery.
            </p>
          </div>

          <div className="rb-3col">
            <div>
              <h3>Prioritize faster</h3>
              <p>Turn long keyword lists into clear tiers so your team knows what deserves attention first and what can wait.</p>
            </div>
            <div>
              <h3>Plan for modern search</h3>
              <p>Go beyond classic keyword metrics by bringing prompt visibility, mentions, and citations into the same planning workflow.</p>
            </div>
            <div>
              <h3>Create cleaner handoffs</h3>
              <p>Export outputs your team can move directly into content planning, page mapping, and execution without extra cleanup.</p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2 */}
      <section className="rb-sectionSoft">
        <div className="rb-band tight">
          <div className="rb-sectionHeader">
            <div className="rb-kicker">What RankedBox helps you do</div>
            <h2 className="rb-h2">One workflow for keyword decisions and AI discovery visibility</h2>
          </div>

          <div className="rb-2colLines">
            <div className="rb-lineItem">
              <h3>Score and tier opportunities</h3>
              <p>Quickly evaluate search volume and keyword difficulty so teams can separate high-value targets from noise.</p>
            </div>
            <div className="rb-lineItem">
              <h3>Align teams around what to target</h3>
              <p>Give strategists, writers, and stakeholders a clearer view of what matters most so execution stays focused.</p>
            </div>
            <div className="rb-lineItem">
              <h3>Evaluate prompt visibility, mentions, and citations</h3>
              <p>Build strategy around how discovery is changing by looking at the signals that influence how brands appear in AI-assisted search experiences.</p>
            </div>
            <div className="rb-lineItem">
              <h3>Move from analysis to action</h3>
              <p>Export cleaner outputs and use them to support prioritization, mapping decisions, and a more repeatable workflow across client accounts.</p>
            </div>
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="rb-sectionWhite">
        <div className="rb-band">
          <h2 className="rb-finalTitle">Start with the free scorer. Build from there.</h2>
          <p className="rb-finalCopy">
            Use the public scorer to evaluate opportunities fast. When you are ready for a deeper workflow,
            sign in and move into the full RankedBox experience.
          </p>

          <div className="rb-finalActions">
            <Link href="/try" className="rb-btnLight">Try the free scorer</Link>
            <Link href="/pricing" className="rb-btnLight">View pricing</Link>
          </div>
        </div>
      </section>
    </main>
  );
}