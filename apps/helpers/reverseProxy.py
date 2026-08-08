import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

# Hop-by-hop headers are connection-scoped and must not be forwarded to the client:
# passing upstream's Content-Length/Transfer-Encoding through alongside a re-chunked
# StreamingResponse produces a response whose framing contradicts its own body.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
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
        self.client = httpx.AsyncClient(timeout=None)

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
