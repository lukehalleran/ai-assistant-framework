#!/usr/bin/env python3
"""Re-authenticate Google OAuth (Gmail + Calendar + Contacts).

Run this when the stored token has been revoked or expired and automatic refresh
fails. Symptom in the logs:

    [GoogleAuth] Token refresh failed: ('invalid_grant: Token has been expired
    or revoked.', ...)

What it does:
  1. Loads the configured GoogleAuthManager (needs GOOGLE_CLIENT_ID /
     GOOGLE_CLIENT_SECRET in the environment or config.local.yaml).
  2. Backs up the existing token to <token>.bak (nothing is deleted).
  3. Opens your browser for Google consent and saves a fresh token.
  4. Prints the granted scopes for confirmation.

Usage (run in your own terminal so the browser can open):
    python scripts/reauth_google.py
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> int:
    from core.actions.google_auth import get_google_auth

    auth = get_google_auth()
    if auth is None or not auth.is_configured:
        print("❌ Google auth is not configured.")
        print("   Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (env vars or")
        print("   config/config.local.yaml) and try again.")
        return 1

    token_path = Path(auth._token_path)
    if token_path.exists():
        backup = token_path.with_suffix(token_path.suffix + ".bak")
        backup.write_bytes(token_path.read_bytes())
        print(f"• Backed up existing token → {backup}")

    print("• Opening your browser for Google consent…")
    print("  Requesting scopes:")
    for scope in auth._scopes:
        print(f"    - {scope}")

    ok = await auth.authenticate()
    if not ok:
        print("❌ Authentication failed — the token backup (if any) is intact.")
        return 1

    print("✅ Re-authentication successful.")
    creds = auth.get_credentials()
    if creds is not None and getattr(creds, "scopes", None):
        print("  Granted scopes:")
        for scope in creds.scopes:
            print(f"    - {scope}")
    print(f"  Token saved to {token_path}")
    print("  Restart the app (or it will pick up the new token on next launch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
