"""Pluggable backend dispatch for scverse host libraries."""

from __future__ import annotations

from scverse_backends._dispatcher import BackendDispatcher
from scverse_backends._settings import Settings

__all__ = ["BackendDispatcher", "Settings"]
