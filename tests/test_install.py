from __future__ import annotations

import json
import sys
from pathlib import Path

from nesift import installer


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(installer.Path, "home", staticmethod(lambda: tmp_path))


def test_detect_targets_empty(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    assert installer.detect_targets() == []


def test_detect_targets_openclaw(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    (tmp_path / ".openclaw").mkdir()
    assert installer.detect_targets() == ["openclaw"]


def test_install_claude(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    results = installer.install(["claude"], force=False, dry_run=False)
    assert len(results) == 1
    result = results[0]
    assert not result.skipped
    out = tmp_path / ".claude/skills/nesift/SKILL.md"
    assert out.exists()
    assert out.read_text(encoding="utf-8") == (
        installer._SKILLS_SRC / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_install_openclaw(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    results = installer.install(["openclaw"], force=False, dry_run=False)
    assert len(results) == 1
    result = results[0]
    assert not result.skipped
    assert (tmp_path / ".openclaw/skills/nesift/SKILL.md").exists()
    bridge = tmp_path / ".openclaw/skills/nesift/scripts/nesift.py"
    assert bridge.exists()
    assert bridge.read_text(encoding="utf-8") == (
        installer._SKILLS_SRC / "openclaw_bridge.py"
    ).read_text(encoding="utf-8")


def test_install_skip_existing(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    dest = tmp_path / ".claude/skills/nesift"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("existing\n", encoding="utf-8")
    results = installer.install(["claude"], force=False, dry_run=False)
    assert results == [
        installer.InstallResult(
            target="claude",
            dest=dest,
            files_written=[],
            skipped=True,
            reason="already exists: SKILL.md",
        )
    ]
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "existing\n"


def test_install_force(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    dest = tmp_path / ".claude/skills/nesift"
    dest.mkdir(parents=True)
    existing = dest / "SKILL.md"
    existing.write_text("old\n", encoding="utf-8")
    results = installer.install(["claude"], force=True, dry_run=False)
    assert len(results) == 1
    assert not results[0].skipped
    assert existing.read_text(encoding="utf-8") != "old\n"


def test_install_dry_run(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    results = installer.install(["openclaw"], force=False, dry_run=True)
    assert len(results) == 1
    assert not results[0].skipped
    assert results[0].reason == "dry-run"
    assert not (tmp_path / ".openclaw/skills/nesift").exists()


def test_install_unknown_target(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    results = installer.install(["bogus"], force=False, dry_run=False)
    assert results == [
        installer.InstallResult(
            target="bogus",
            dest=tmp_path,
            files_written=[],
            skipped=True,
            reason="unknown target",
        )
    ]


def test_install_mcp_not_found(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    ok, message = installer.install_mcp(dry_run=False)
    assert ok is False
    assert "nesift-mcp not found in PATH" in message
