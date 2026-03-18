"""
Supabase validator

Looks up a student ID against a Supabase `students` table and returns the
registered name.  Disabled automatically when SUPABASE_ENABLED = False in
config.py, so the rest of the pipeline works without any Supabase setup.

Expected table schema:
    students (
        student_id  TEXT PRIMARY KEY,   -- e.g. "21234567"
        name        TEXT NOT NULL        -- e.g. "Conor Clancy"
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


def lookup_student(student_id):
    """
    Query Supabase for a student by their ID number.

    Args:
      student_id: string of digits, e.g. "21234567"

    Returns:
      (found: bool, name: str | None)
      Returns (False, None) when Supabase is disabled or the ID is not in the table.
    """
    if not config.SUPABASE_ENABLED or not student_id:
        return False, None

    client = _get_client()
    if client is None:
        return False, None

    try:
        result = (
            client.table("students")
            .select("name")
            .eq("student_id", student_id)
            .single()
            .execute()
        )
        if result.data:
            return True, result.data.get("name")
        return False, None
    except Exception as e:
        print(f"[Supabase] Lookup failed for ID {student_id}: {e}")
        return False, None
