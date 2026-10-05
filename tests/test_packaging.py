"""Packaging guard: what this public repository may and may not contain.

The manuscript, its build toolchain and the submission correspondence live beside the repository
but must never be published from it. Two views are checked, because they differ: what git would
publish (tracked plus untracked-but-not-ignored files), and what a zip of the working tree would
hold. A guard that only asked git would pass while a forbidden file sat in the folder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("MANUSCRIPT", "PAPER_OUTLINE", "references", "submission/", "src/",
                      "CLAUDE.md", "package.json", "package-lock.json")
TEXT = (".py", ".md", ".cff", ".txt", ".toml", ".cfg", ".yml", ".yaml", ".json", ".csv")
EM_DASH = chr(0x2014)


def publishable() -> list[str]:
    """Files git would publish: tracked, plus untracked files that are not ignored."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.splitlines() if (ROOT / f).exists()]


def test_no_private_files_would_be_published():
    bad = [f for f in publishable() if f.startswith(FORBIDDEN_PREFIXES)]
    assert not bad, f"private files would be published: {bad}"


def test_private_files_are_ignored_when_present():
    """The working-tree view: anything private in the folder must be covered by .gitignore."""
    present = [p for p in ROOT.iterdir() if p.name.startswith(FORBIDDEN_PREFIXES)
               or p.name in ("submission", "src")]
    for p in present:
        r = subprocess.run(["git", "check-ignore", "-q", p.name], cwd=ROOT)
        assert r.returncode == 0, f"{p.name} is in the folder and not ignored"


def test_no_em_dashes_in_published_text():
    bad = [f for f in publishable() if f.endswith(TEXT)
           and EM_DASH in (ROOT / f).read_text(encoding="utf-8", errors="ignore")]
    assert not bad, f"em dashes in: {bad}"
