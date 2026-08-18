"""Tests for ReverseProxy's streaming pass-through.

The proxy used to aread() the whole upstream response into memory and copy it into
a Response, so every payload was resident several times per in-flight request; it
also leaked the upstream connection when aread() raised part-way through (httpx
closes a streamed response only at the end of a fully-consumed iteration, which a
client navigating away mid-load never reaches).
"""

import httpx
import pytest
from fastapi import FastAPI, Request

from apps.helpers.reverseProxy import ReverseProxy


def _upstream_app(handler) -> httpx.ASGITransport:
    app = FastAPI()
    app.add_api_route("/{path:path}", handler, methods=["GET", "POST"])
    return httpx.ASGITransport(app=app)


def _proxy_to(transport: httpx.ASGITransport) -> ReverseProxy:
    proxy = ReverseProxy(base_url="http://upstream")
    proxy.client = httpx.AsyncClient(transport=transport, timeout=None)
    return proxy


def _client_for(proxy: ReverseProxy) -> httpx.AsyncClient:
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def _fwd(path: str, request: Request):
        return await proxy.handle_request(path, request)

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class TestBodyPassThrough:
    async def test_response_body_is_forwarded_intact(self):
        payload = b"\x89PNG" + b"x" * 5000

        async def handler(path: str):
            from fastapi.responses import Response

            return Response(content=payload, media_type="image/png")

        proxy = _proxy_to(_upstream_app(handler))
        async with _client_for(proxy) as client:
            res = await client.get("/img.png")

        assert res.status_code == 200
        assert res.content == payload
        assert res.headers["content-type"] == "image/png"

    async def test_status_code_is_preserved(self):
        async def handler(path: str):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "nope"}, status_code=404)

        proxy = _proxy_to(_upstream_app(handler))
        async with _client_for(proxy) as client:
            res = await client.get("/missing")

        assert res.status_code == 404
        assert res.json() == {"detail": "nope"}

    async def test_request_body_and_query_are_forwarded(self):
        seen = {}

        async def handler(path: str, request: Request):
            seen["body"] = await request.body()
            seen["query"] = str(request.query_params)
            return {"ok": True}

        proxy = _proxy_to(_upstream_app(handler))
        async with _client_for(proxy) as client:
            await client.post("/thing?a=1&b=2", content=b"hello-body")

        assert seen["body"] == b"hello-body"
        assert seen["query"] == "a=1&b=2"


class TestRangeRequests:
    """What makes an event clip scrubbable through the SPA's single port.

    The NVR event review modal plays `/api/recordings/{uuid}/file` via a `<video>`
    element, which seeks by issuing Range requests. That only works if the proxy
    forwards the `Range` header up and returns the upstream's 206 and `Content-Range`
    back down — otherwise the video plays from the start and the scrubber does nothing
    but look functional. The alternative would be pointing the browser at db_server's
    port directly, which breaks the "the SPA talks only to 7999" rule.
    """

    @staticmethod
    def _range_upstream(payload: bytes):
        async def handler(path: str, request: Request):
            from fastapi.responses import Response

            header = request.headers.get("range")
            if not header:
                return Response(content=payload, media_type="video/mp4")
            first, _, last = header.removeprefix("bytes=").partition("-")
            start = int(first)
            end = int(last) if last else len(payload) - 1
            return Response(
                content=payload[start : end + 1],
                status_code=206,
                media_type="video/mp4",
                headers={
                    "content-range": f"bytes {start}-{end}/{len(payload)}",
                    "accept-ranges": "bytes",
                },
            )

        return handler

    async def test_range_header_reaches_upstream_and_206_comes_back(self):
        payload = bytes(range(256)) * 40
        proxy = _proxy_to(_upstream_app(self._range_upstream(payload)))

        async with _client_for(proxy) as client:
            res = await client.get("/clip.mp4", headers={"Range": "bytes=100-199"})

        assert res.status_code == 206
        assert res.content == payload[100:200]
        assert res.headers["content-range"] == f"bytes 100-199/{len(payload)}"

    async def test_accept_ranges_survives_so_the_browser_knows_it_can_seek(self):
        payload = b"\x00" * 1000
        proxy = _proxy_to(_upstream_app(self._range_upstream(payload)))

        async with _client_for(proxy) as client:
            res = await client.get("/clip.mp4", headers={"Range": "bytes=0-9"})

        assert res.headers["accept-ranges"] == "bytes"

    async def test_a_rangeless_request_still_gets_the_whole_file(self):
        payload = b"\x00\x01" * 500
        proxy = _proxy_to(_upstream_app(self._range_upstream(payload)))

        async with _client_for(proxy) as client:
            res = await client.get("/clip.mp4")

        assert res.status_code == 200
        assert res.content == payload


class TestHopByHopHeaders:
    async def test_upstream_content_length_is_not_forwarded(self):
        """A re-chunked StreamingResponse must not carry upstream's framing headers."""

        async def handler(path: str):
            from fastapi.responses import Response

            return Response(content=b"12345", media_type="text/plain")

        proxy = _proxy_to(_upstream_app(handler))
        async with _client_for(proxy) as client:
            res = await client.get("/x")

        assert res.content == b"12345"
        # Starlette re-derives framing for the streamed body; the upstream value must
        # not have been copied through verbatim alongside it.
        assert "transfer-encoding" not in {k.lower() for k in res.headers} or "content-length" not in {
            k.lower() for k in res.headers
        }

    async def test_non_hop_by_hop_headers_survive(self):
        async def handler(path: str):
            from fastapi.responses import Response

            return Response(content=b"{}", media_type="application/json", headers={"x-custom": "kept"})

        proxy = _proxy_to(_upstream_app(handler))
        async with _client_for(proxy) as client:
            res = await client.get("/x")

        assert res.headers["x-custom"] == "kept"


class TestUpstreamResponseIsClosed:
    async def test_response_is_closed_after_a_full_read(self):
        closed = []

        async def handler(path: str):
            from fastapi.responses import Response

            return Response(content=b"body", media_type="text/plain")

        proxy = _proxy_to(_upstream_app(handler))
        real_send = proxy.client.send

        async def _tracking_send(*args, **kwargs):
            res = await real_send(*args, **kwargs)
            real_close = res.aclose

            async def _close():
                closed.append(True)
                await real_close()

            res.aclose = _close
            return res

        proxy.client.send = _tracking_send

        async with _client_for(proxy) as client:
            await client.get("/x")

        assert closed, "upstream response must be closed once the body is consumed"

    async def test_response_is_closed_when_the_body_iteration_is_abandoned(self):
        """The leak path: client goes away mid-body, so the stream is never drained."""
        closed = []

        async def handler(path: str):
            from fastapi.responses import Response

            return Response(content=b"y" * 100_000, media_type="application/octet-stream")

        proxy = _proxy_to(_upstream_app(handler))
        real_send = proxy.client.send

        async def _tracking_send(*args, **kwargs):
            res = await real_send(*args, **kwargs)
            real_close = res.aclose

            async def _close():
                closed.append(True)
                await real_close()

            res.aclose = _close
            return res

        proxy.client.send = _tracking_send

        async with _client_for(proxy) as client:
            async with client.stream("GET", "/big") as res:
                await res.aiter_raw().__anext__()  # read one chunk, then bail

        assert closed, "abandoning the body must still close the upstream response"


class TestUrlConstruction:
    @pytest.mark.parametrize(
        "base_path,req_path,expected_suffix",
        [
            ("", "thing", "/thing"),
            ("/api", "thing", "/api/thing"),
            ("api", "/thing", "/api/thing"),
        ],
    )
    async def test_paths_join_without_duplicate_slashes(self, base_path, req_path, expected_suffix):
        seen = {}

        async def handler(path: str, request: Request):
            seen["path"] = request.url.path
            return {"ok": True}

        proxy = ReverseProxy(base_url="http://upstream", path=base_path)
        proxy.client = httpx.AsyncClient(transport=_upstream_app(handler), timeout=None)

        async with _client_for(proxy) as client:
            await client.get(f"/{req_path.lstrip('/')}")

        assert seen["path"] == expected_suffix


class TestTimeouts:
    """The proxy is built with timeout=None everywhere, which also disables the
    *pool* timeout: once the connection pool is saturated, every further proxied
    request waits for a free slot forever instead of failing. The browser shows a
    request that never resolves and no error is ever logged.

    Read must stay unbounded — SSE streams (/api/tasks/stream, /api/notices/stream)
    and large image downloads are legitimately long-lived — but connecting,
    writing, and waiting for a pool slot are all bounded operations.
    """

    def test_read_timeout_stays_unbounded_for_sse(self):
        timeout = ReverseProxy(base_url="http://upstream").client.timeout
        assert timeout.read is None, "a finite read timeout would sever long-lived SSE streams"

    def test_pool_and_connect_timeouts_are_finite(self):
        timeout = ReverseProxy(base_url="http://upstream").client.timeout
        assert timeout.pool is not None, "an unbounded pool timeout hangs forever once the pool saturates"
        assert timeout.connect is not None, "an unbounded connect timeout hangs on an unreachable upstream"
        assert timeout.write is not None
