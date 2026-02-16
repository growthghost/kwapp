// web/src/app/pricing/page.js
export default function PricingPage() {
  return (
    <main className="rb-container">
      <h1 className="rb-title">Pricing</h1>
      <p className="rb-subtitle">
        Simple, transparent plans. (Placeholder for now.)
      </p>

      <div className="rb-card" style={{ marginTop: 18, maxWidth: 820 }}>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>
          Coming soon
        </div>
        <div style={{ color: "var(--muted)" }}>
          Next: add 2–3 pricing tiers + a FAQ section.
        </div>
      </div>
    </main>
  );
}
