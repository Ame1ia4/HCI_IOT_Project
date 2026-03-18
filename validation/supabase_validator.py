"""
Supabase scan logger

Writes a row to the `scans` table on every validation result.
No PII is stored — only a UUID, timestamp, and pass/fail result.

Expected table schema:
    scans (
        id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at TIMESTAMPTZ      DEFAULT now(),
        is_valid   BOOLEAN          NOT NULL
    )
"""

import config

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not config.SUPABASE_ENABLED:
        return None
    try:
        from supabase import create_client
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    except Exception as e:
        print(f"[Supabase] Failed to connect: {e}")
    return _client


def log_scan(is_valid):
    """
    Insert a scan event into the scans table.

    Args:
      is_valid: bool — whether the card passed validation.
    """
    if not config.SUPABASE_ENABLED:
        return

    client = _get_client()
    if client is None:
        return

    try:
        client.table("scans").insert({"is_valid": is_valid}).execute()
    except Exception as e:
        print(f"[Supabase] Failed to log scan: {e}")
