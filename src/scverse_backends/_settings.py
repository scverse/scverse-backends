"""Per-host backend settings backed by a thread-safe ContextVar."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from scverse_backends._registry import _Registry


class _Settings:
    """Per-host settings exposing ``.backend`` and ``.use_backend()``.

    Each ``BackendDispatcher`` owns one ``_Settings`` with its own
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

        Set to a registered backend name or alias (e.g. ``'gpu'``, ``'cuda'``).
        Aliases are resolved to the canonical name on assignment.
        """
        return self._backend_var.get()

    @backend.setter
    def backend(self, value: str) -> None:
        if value == "cpu":
            self._backend_var.set(value)
            return

        canonical = self._registry.resolve_name(value)

        # Completely unknown name — suggest alternatives
        if canonical is None:
            raise ValueError(self._registry.suggest(value))

        # Trusted but not installed
        if self._registry.is_trusted(canonical) and self._registry.get_backend(canonical) is None:
            package = self._registry.trusted_package(canonical) or canonical
            raise ImportError(
                f"Backend {value!r} ({canonical}) is not installed. Install it with: pip install {package}"
            )

        # Known alias but backend not loaded
        if self._registry.get_backend(canonical) is None:
            raise ImportError(f"Backend {value!r} is not installed.")

        # Warn if untrusted
        self._registry.check_trusted(canonical)

        # Always store the canonical name
        self._backend_var.set(canonical)

    @contextmanager
    def use_backend(self, backend: str) -> Generator[None, None, None]:
        """Temporarily set the backend within a context.

        Examples
        --------
        >>> with settings.use_backend("gpu"):
        ...     ...
        """
        token = self._backend_var.set(self.backend)
        try:
            self.backend = backend
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
