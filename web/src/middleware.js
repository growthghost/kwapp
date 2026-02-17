// web/src/middleware.js
import { NextResponse } from "next/server";

export function middleware(req) {
  const session = req.cookies.get("rb_session")?.value;

  // Protect /app (and anything under it)
  if (!session) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", req.nextUrl.pathname);
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/app/:path*"],
};
