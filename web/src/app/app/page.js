// web/src/app/app/page.js
export default function AppPage() {
  return (
    <main className="rb-container">
      <h1 className="rb-title">App</h1>
      <p className="rb-subtitle">
        This area will be gated behind login. For now, this is a placeholder page.
      </p>

      <div className="rb-card" style={{ marginTop: 18, maxWidth: 820 }}>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>
          Coming soon
        </div>
        <div style={{ color: "var(--muted)" }}>
          Next: login gate → then connect this page to your secure Python API.
        </div>
      </div>
    </main>
  );
}
