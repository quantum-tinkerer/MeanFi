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


def test_repo_uses_unified_performance_layout():
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

    assert (repo_root / "examples").is_dir()
    assert (repo_root / "examples" / "demo.py").is_file()
    assert (repo_root / "performance").is_dir()
    assert (repo_root / "performance" / "benchmarks").is_dir()
    assert (repo_root / "performance" / "profiling").is_dir()
    assert not any(path == "benchmarks" or path.startswith("benchmarks/") for path in tracked)
    assert not any(path == "demo.py" or path.startswith("demo.py/") for path in tracked)
    assert not any(path == "profiling" or path.startswith("profiling/") for path in tracked)
