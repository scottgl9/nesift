import httpx
import pytest
import respx

from nesift.fetcher import FetchError, fetch


@respx.mock
def test_fetch_success():
    respx.get("https://example.test/").mock(
        return_value=httpx.Response(200, text="<html><body>ok</body></html>")
    )
    assert "ok" in fetch("https://example.test/")


@respx.mock
def test_fetch_http_error():
    respx.get("https://example.test/").mock(return_value=httpx.Response(404, text="nope"))
    with pytest.raises(FetchError):
        fetch("https://example.test/")


@respx.mock
def test_fetch_network_error():
    respx.get("https://example.test/").mock(side_effect=httpx.ConnectError("dns"))
    with pytest.raises(FetchError):
        fetch("https://example.test/")
