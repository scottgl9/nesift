import httpx
import pytest
import respx

from nesift.searxng import SearxNGError, search


@respx.mock
def test_search_parses_results():
    respx.get("http://my.searx/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a", "content": "snippet a", "score": 1.0},
                    {"title": "B", "url": "https://b", "content": "snippet b"},
                ]
            },
        )
    )
    out = search("hello", top_n=5, instance_url="http://my.searx")
    assert len(out) == 2
    assert out[0].url == "https://a"
    assert out[0].snippet == "snippet a"


@respx.mock
def test_search_top_n_truncates():
    items = [
        {"title": str(i), "url": f"https://x/{i}", "content": "..."} for i in range(10)
    ]
    respx.get("http://my.searx/search").mock(
        return_value=httpx.Response(200, json={"results": items})
    )
    out = search("q", top_n=3, instance_url="http://my.searx")
    assert len(out) == 3


@respx.mock
def test_search_http_error_raises():
    respx.get("http://my.searx/search").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    with pytest.raises(SearxNGError):
        search("q", instance_url="http://my.searx")


@respx.mock
def test_search_network_error_raises():
    respx.get("http://my.searx/search").mock(side_effect=httpx.ConnectError("nope"))
    with pytest.raises(SearxNGError):
        search("q", instance_url="http://my.searx")
