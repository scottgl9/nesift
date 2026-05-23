from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nesift import cli as cli_mod

runner = CliRunner()


def test_init_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(cli_mod.app, ["init", "--file", "AGENTS.md"])
    assert res.exit_code == 0
    out = Path(tmp_path / "AGENTS.md").read_text()
    assert "nesift" in out
    assert "nesift query" in out


def test_init_appends_when_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# Existing\n\nSome content.\n")
    res = runner.invoke(cli_mod.app, ["init"])
    assert res.exit_code == 0
    out = (tmp_path / "AGENTS.md").read_text()
    assert "Existing" in out and "nesift" in out


def test_init_skips_when_already_mentions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Use nesift for research.\n")
    res = runner.invoke(cli_mod.app, ["init"])
    assert res.exit_code == 0
    assert "already mentions" in res.stdout
