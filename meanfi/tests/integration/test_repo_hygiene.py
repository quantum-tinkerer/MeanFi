from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.integration


def test_repo_does_not_track_generated_root_build_artifacts():
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".git").exists():
        pytest.skip("git metadata is unavailable")

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = set(completed.stdout.splitlines())
    forbidden = {
        ".skbuild-info.json",
        "build.ninja",
        "cmake_install.cmake",
    }

    assert tracked.isdisjoint(forbidden)