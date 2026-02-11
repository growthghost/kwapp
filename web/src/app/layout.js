import "./globals.css";

export const metadata = {
  title: "RankedBox",
  description: "Score keywords by Search Volume and Keyword Difficulty",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <header className="rbTopbar">
          <div className="rbBrand">
            <img
              className="rbLogo"
              src="/RankedBox Logo Icon.png"
              alt="RankedBox"
            />
            <span className="rbBrandText">RankedBox</span>
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}
