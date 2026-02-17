import { NextResponse } from "next/server";

export function middleware(req) {
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/app/:path*"],
};
