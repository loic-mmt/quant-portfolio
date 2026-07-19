from __future__ import annotations

from datetime import datetime, timezone
import re
import secrets


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def validate_run_id(run_id: str) -> str:
    """Validate an identifier before it is used in a path or partition."""
    value = str(run_id).strip()
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must contain 1-96 letters, digits, dots, underscores or hyphens"
        )
    return value


def create_run_id(prefix: str = "run", now: datetime | None = None) -> str:
    """Create a sortable UTC run identifier with a small collision suffix."""
    clean_prefix = validate_run_id(prefix)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    value = f"{clean_prefix}-{timestamp:%Y%m%dT%H%M%S%fZ}-{secrets.token_hex(3)}"
    return validate_run_id(value)


def ensure_run_id(run_id: str | None, prefix: str = "run") -> str:
    """Return a validated caller-provided id or generate a new one."""
    return validate_run_id(run_id) if run_id is not None else create_run_id(prefix)
