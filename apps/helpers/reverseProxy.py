import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

# Hop-by-hop headers are connection-scoped and describe one link, so forwarding them
# would describe this connection with the framing of the upstream one.
#
# `content-length` and `content-encoding` are deliberately NOT in here. The body is
# forwarded byte-for-byte off `aiter_raw()`, so upstream's length and encoding both
# still describe it exactly — and dropping the length is what turned every clip
# response into a chunked one of unknown size. Firefox will not resume a suspended
# `preload="metadata"` load without a length, so a clip played one frame and stalled
# while seeking, which re-requests a fresh range each time, kept working.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


class ReverseProxy:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        path: str = "",
        *,
        base_url: str = None,
    ):
        path_part = path if not path or path.startswith("/") else f"/{path}"
        if base_url is not None:
            self.target_base_url = base_url.rstrip("/") + path_part
        else:
            self.target_base_url = f"http://{host}:{port}{path_part}"
        # read=None because SSE streams (/api/tasks/stream, /api/notices/stream) and
        # large image downloads are legitimately long-lived. Everything else is
        # bounded: a blanket timeout=None also disables the *pool* timeout, so once
        # the pool saturates — leaked upstream connections pile up as CLOSE_WAIT —
        # every later request waits for a slot forever. That surfaces as a browser
        # request that never resolves, with nothing logged to explain it.
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10.0, write=30.0, pool=30.0),
        )

    async def handle_request(self, path: str, request: Request):
        url = f"{self.target_base_url}"
        path = path if not path or path.startswith("/") else f"/{path}"
        url += path

        if request.query_params:
            url += f"?{request.query_params}"

        proxy_req = self.client.build_request(
            method=request.method,
            url=url,
            headers=request.headers.raw,
            content=await request.body(),
        )

        proxy_res = await self.client.send(proxy_req, stream=True)

        # Stream the body straight through rather than aread()-ing it into memory.
        # The previous version buffered the whole upstream response, then copied it
        # again into the outgoing Response — for image traffic that meant several
        # copies of every payload resident per in-flight request. It also leaked the
        # upstream connection whenever aread() raised part-way (httpx only closes the
        # response at the *end* of a fully-consumed iteration), which is exactly what
        # happens when a client navigates away mid-load.
        async def body_iter():
            try:
                async for chunk in proxy_res.aiter_raw():
                    yield chunk
            finally:
                await proxy_res.aclose()

        headers = {k: v for k, v in proxy_res.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}

        return StreamingResponse(
            body_iter(),
            status_code=proxy_res.status_code,
            headers=headers,
        )
