"""Shared types"""
from typing import Optional

from pydantic import BaseModel

from .consts import JsonValues


class StatusResponse(BaseModel):
    status: str = JsonValues.SUCCESS
    message: Optional[str] = None
