"""Provider-agnostic email layer (2026-09-01) — see docs/EMAIL_INTEGRATION_DESIGN.md.

Read-only, metadata-first, live-fetch-only. Public surface:
`core.email.service.get_email_service()` + `core.email.provider.EmailMessage`.
"""
