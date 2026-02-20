// web/src/app/layout.js
import "./globals.css";
import Link from "next/link";
import HeaderNav from "./components/HeaderNav";

export const metadata = {
  title: "RANKEDBOX",
  description: "Score keywords by Search Volume and Keyword Difficulty",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <header className="rb-header">
          <div className="rb-header-inner">
            <Link href="/" className="rb-brand" aria-label="Go to home">
              <img
                className="rb-logo"
                src="/RankedBoxLogoIcon.png"
                alt="RankedBox"
                width={24}
                height={24}
                style={{ width: 24, height: 24, objectFit: "contain", display: "block" }}
              />
              <span className="rb-brand-name">RANKEDBOX</span>
            </Link>

            <HeaderNav />
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}