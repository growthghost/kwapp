export default function TryPage() {
  return (
    <main style={{ padding: 24 }}>
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>Try RankedBox</h1>
      <p style={{ marginBottom: 24 }}>
        This is the public /try page. We’ll put the single keyword score tool here next.
      </p>

      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          maxWidth: 720,
        }}
      >
        <h2 style={{ fontSize: 20, marginBottom: 12 }}>Single Keyword Score</h2>

        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: 6 }}>Search Volume (A)</label>
            <input
              type="number"
              placeholder="50"
              style={{
                width: "100%",
                padding: 10,
                borderRadius: 10,
                border: "1px solid #d1d5db",
              }}
            />
          </div>

          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: 6 }}>Keyword Difficulty (B)</label>
            <input
              type="number"
              placeholder="21"
              style={{
                width: "100%",
                padding: 10,
                borderRadius: 10,
                border: "1px solid #d1d5db",
              }}
            />
          </div>
        </div>

        <button
          style={{
            background: "#000000",
            color: "#ffffff",
            padding: "10px 14px",
            borderRadius: 10,
            border: "none",
            cursor: "pointer",
          }}
        >
          Calculate Score
        </button>
      </div>
    </main>
  );
}
