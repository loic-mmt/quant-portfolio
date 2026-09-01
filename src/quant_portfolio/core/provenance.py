"""Deterministic code and Git provenance for reproducible research runs."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from quant_portfolio.core.settings import PROJECT_ROOT

SOURCE_ROOTS = ("src", "config", "sql", "pyproject.toml", "requirements.lock")


def source_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Hash research code/config, including untracked files under known roots."""
    paths: list[Path] = []
    for name in SOURCE_ROOTS:
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(args: list[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_provenance(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Capture UTC generation time, Git revision/dirty state, and source hash."""
    revision = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=all"], root)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "git_dirty": None if status is None else bool(status),
        "source_fingerprint": source_fingerprint(root),
    }
