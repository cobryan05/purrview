"""Shared types"""
from pydantic import BaseModel

from .consts import JsonValues


class StatusResponse(BaseModel):
    status: str = JsonValues.SUCCESS
    message: str | None = None
