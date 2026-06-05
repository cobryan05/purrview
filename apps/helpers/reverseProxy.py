import httpx
from fastapi import Request, Response


class ReverseProxy:
    def __init__(self, host: str = None, port: int = None, path: str = "", *, base_url: str = None):
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
            method=request.method, url=url, headers=request.headers.raw, content=await request.body()
        )

        proxy_res = await self.client.send(proxy_req, stream=True)

        return Response(
            content=await proxy_res.aread(), status_code=proxy_res.status_code, headers=dict(proxy_res.headers)
        )
