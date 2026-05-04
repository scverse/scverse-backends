"""The ``BackendDispatcher`` class — a host's full dispatch surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scverse_backends._dispatch import _Dispatch
from scverse_backends._registry import _Registry
from scverse_backends._settings import _Settings

if TYPE_CHECKING:
    from collections.abc import Callable


class BackendDispatcher:
    """Per-host dispatch surface.

    Each scverse host library (squidpy, scanpy, ...) instantiates one
    ``BackendDispatcher`` and re-exports its attributes as the host's
    public dispatch API.

    Parameters
    ----------
    entrypoint_group
        Entrypoint group that backend packages register against
        (e.g. ``"squidpy.backends"``).
    host_name
        Name of the host library, used in error messages.
    trusted_backends
        Mapping of canonical backend name to ``{"aliases": [...], "package": "pip-name"}``.
        Backends not in this list still work but emit a warning on first use.

    Examples
    --------
    >>> from scverse_backends import BackendDispatcher
    >>> _dispatcher = BackendDispatcher(
    ...     entrypoint_group="squidpy.backends",
    ...     host_name="squidpy",
    ...     trusted_backends={
    ...         "rapids_singlecell": {
    ...             "aliases": ["rsc", "cuda", "gpu"],
    ...             "package": "rapids-singlecell",
    ...         },
    ...     },
    ... )
    >>> dispatch = _dispatcher.dispatch
    >>> settings = _dispatcher.settings
    """

    def __init__(
        self,
        *,
        entrypoint_group: str,
        host_name: str,
        trusted_backends: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.entrypoint_group = entrypoint_group
        self.host_name = host_name
        self._registry = _Registry(
            entrypoint_group=entrypoint_group,
            host_name=host_name,
            trusted_backends=trusted_backends or {},
        )
        self._settings = _Settings(self._registry)
        self._dispatch_impl = _Dispatch(self._registry, self._settings)

    @property
    def dispatch(self) -> Callable:
        """The ``@dispatch`` decorator for host functions."""
        return self._dispatch_impl.decorator

    @property
    def settings(self) -> _Settings:
        """Settings object exposing ``.backend`` and ``.use_backend()``."""
        return self._settings

    def get_backend(self, name: str) -> Any | None:
        """Look up a backend by name or alias. Returns ``None`` for ``"cpu"``."""
        return self._registry.get_backend(name)

    def available_backend_names(self) -> list[str]:
        """All registered backend names and aliases."""
        return self._registry.available_backend_names()
