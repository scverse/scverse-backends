"""Per-host backend settings backed by a thread-safe ContextVar."""

from __future__ import annotations

from collections.abc import Generator  # noqa: TC003
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from scverse_backends._registry import _Registry  # noqa: TC001


class Settings:
    """Per-host settings exposing ``.backend`` and ``.use_backend()``.

    Each ``BackendDispatcher`` owns one ``Settings`` with its own
    ``ContextVar`` so host libraries' active backends are isolated.
    """

    def __init__(self, registry: _Registry) -> None:
        self._registry = registry
        # Each instance gets its own ContextVar; the var name is purely for
        # debugging/repr — the actual storage is per-instance.
        self._backend_var: ContextVar[str] = ContextVar(
            f"{registry.host_name}_backend",
            default="cpu",
        )

    @property
    def backend(self) -> str:
        """The active backend name (default ``'cpu'``).

        Set to a registered backend name or alias (e.g. ``'cuda'``).
        Aliases are resolved to the canonical name on assignment.
        """
        return self._backend_var.get()

    @backend.setter
    def backend(self, value: str) -> None:
        canonical, _ = self._registry.require_backend(value)
        self._backend_var.set(canonical)

    @contextmanager
    def use_backend(self, backend: str) -> Generator[None, None, None]:
        """Temporarily set the backend within a context.

        Examples
        --------
        >>> with settings.use_backend("cuda"):
        ...     ...
        """
        canonical, _ = self._registry.require_backend(backend)
        token = self._backend_var.set(canonical)
        try:
            yield
        finally:
            self._backend_var.reset(token)

    def available_backends(self) -> list[str]:
        """Return canonical names of all discovered backends."""
        self._registry._ensure_discovered()
        return sorted(self._registry._backends.keys())

    def get_backend(self, name: str) -> Any | None:
        """Look up a backend by name or alias."""
        return self._registry.get_backend(name)


_Settings = Settings
