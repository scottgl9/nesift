from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from nesift import cli as cli_mod
from nesift.embedder import FakeEmbedder

runner = CliRunner()


def _patch_embedder(monkeypatch):
    monkeypatch.setattr(cli_mod, "_embedder", lambda fast=False, lang=False: FakeEmbedder(dim=64))


def _session_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    # Force a clean session id so /tmp leakage between tests is impossible.
    monkeypatch.setenv("NESIFT_SESSION", f"test-{tmp_path.name}")


def test_cli_version():
    res = runner.invoke(cli_mod.app, ["version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.stdout


@respx.mock
def test_cli_add_query_clear(monkeypatch, tmp_path, fixture_html):
    _session_env(tmp_path, monkeypatch)
    _patch_embedder(monkeypatch)
    respx.get("https://blog.test/r").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )

    res = runner.invoke(cli_mod.app, ["add", "https://blog.test/r", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["chunks"] > 0

    res = runner.invoke(cli_mod.app, ["list", "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout)[0]["url"] == "https://blog.test/r"

    res = runner.invoke(cli_mod.app, ["query", "exponential backoff", "--json"])
    assert res.exit_code == 0
    body = json.loads(res.stdout)
    assert body["results"]

    res = runner.invoke(cli_mod.app, ["answer", "what is exponential backoff", "--json"])
    assert res.exit_code == 0
    body = json.loads(res.stdout)
    assert "[1]" in body["answer"]

    res = runner.invoke(cli_mod.app, ["clear"])
    assert res.exit_code == 0
    assert "cleared" in res.stdout

    res = runner.invoke(cli_mod.app, ["list", "--json"])
    assert json.loads(res.stdout) == []


def test_cli_score(monkeypatch, tmp_path):
    _session_env(tmp_path, monkeypatch)
    _patch_embedder(monkeypatch)
    res = runner.invoke(
        cli_mod.app,
        [
            "score",
            "vector database",
            "Pinecone is a managed vector database.",
            "How to bake bread.",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    body = json.loads(res.stdout)
    assert body[0]["index"] in (0, 1)
    assert body[0]["score"] >= body[-1]["score"]


@respx.mock
def test_cli_search(monkeypatch, tmp_path, fixture_html):
    _session_env(tmp_path, monkeypatch)
    _patch_embedder(monkeypatch)
    monkeypatch.setenv("NESIFT_SEARXNG_URL", "http://my.searx")
    respx.get("http://my.searx/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Retries",
                        "url": "https://blog.test/r",
                        "content": "retry exponential backoff jitter",
                    }
                ]
            },
        )
    )
    respx.get("https://blog.test/r").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )
    res = runner.invoke(
        cli_mod.app,
        ["search", "retry logic", "--top", "1", "--budget", "800", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    body = json.loads(res.stdout)
    assert body["results"]
    assert body["snippets"]


def test_cli_save(monkeypatch, tmp_path, fixture_html):
    _session_env(tmp_path, monkeypatch)
    _patch_embedder(monkeypatch)
    with respx.mock(assert_all_called=False) as router:
        router.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        runner.invoke(cli_mod.app, ["add", "https://blog.test/r", "--json"])
    out = tmp_path / "snapshot.json"
    res = runner.invoke(cli_mod.app, ["save", "-o", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["pages"]
