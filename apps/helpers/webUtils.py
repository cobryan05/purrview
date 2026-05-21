"""Utility functions for web apis"""

import asyncio
from functools import wraps

from pydantic import BaseModel


def api_forward_request(api, api_method_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(**kwargs):
            payload = kwargs.get("payload", kwargs.get("_payload"))
            if isinstance(payload, BaseModel):
                args = payload.model_dump()
            else:
                args = None
            api_method = getattr(api, api_method_name)
            return await asyncio.to_thread(api_method, args)
        return wrapper
    return decorator
