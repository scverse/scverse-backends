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

    Each host library instantiates one ``BackendDispatcher`` and
    re-exports its attributes as the host's public dispatch API.

    Parameters
    ----------
    entrypoint_group
        Entrypoint group that backend packages register against
        (e.g. ``"example_host.backends"``). Entrypoints may load a
        backend module, class, or object.
    host_name
        Name of the host library, used in error messages.
    trusted_backends
        Mapping of canonical backend name to ``{"aliases": [...], "package": "pip-name"}``.
        Backends not in this list still work but emit a warning on first use.
        For trusted backends, ``package`` is also used as the expected Python
        distribution name during entrypoint discovery. Use ``distributions`` for
        backends that can be provided by multiple distribution names, and
        optionally ``entrypoints``, ``object_refs``, or ``module_prefixes`` for
        stricter provider verification.
    reserved_backends
        Host-owned backend names or aliases that no backend may claim. Values
        are human-readable reasons shown when users request a reserved name.

    Examples
    --------
    >>> from scverse_backends import BackendDispatcher
    >>> _dispatcher = BackendDispatcher(
    ...     entrypoint_group="example_host.backends",
    ...     host_name="example_host",
    ...     trusted_backends={
    ...         "example_accel": {
    ...             "aliases": ["example", "accelerated"],
    ...             "package": "example-host-accel",
    ...         },
    ...     },
    ... )
    >>> backend_dispatch = _dispatcher.backend_dispatch
    >>> settings = _dispatcher.settings
    """

    def __init__(
        self,
        *,
        entrypoint_group: str,
        host_name: str,
        trusted_backends: dict[str, dict[str, Any]] | None = None,
        reserved_backends: dict[str, str] | None = None,
    ) -> None:
        self.entrypoint_group = entrypoint_group
        self.host_name = host_name
        self._registry = _Registry(
            entrypoint_group=entrypoint_group,
            host_name=host_name,
            trusted_backends=trusted_backends or {},
            reserved_backends=reserved_backends,
        )
        self._settings = _Settings(self._registry)
        self._dispatch_impl = _Dispatch(self._registry, self._settings)

    @property
    def backend_dispatch(self) -> Callable:
        """The ``@backend_dispatch`` decorator for host functions."""
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

    def discover(self) -> None:
        """Eagerly load backends and merge their params into host signatures.

        Discovery is normally lazy — entrypoints are loaded the first time
        anything in the dispatcher is queried (settings setter,
        ``get_backend``, a non-CPU dispatched call). Call ``discover()``
        when you want that to happen on a schedule you control:

        * In a host's Sphinx ``conf.py``, so autodoc sees the merged
          signatures with backend-specific params and their docstrings.
        * In tests that introspect ``help(fn)`` / ``inspect.signature(fn)``.
        * Anywhere else IDE/LSP integrations would otherwise see the
          un-merged signature.

        Idempotent — safe to call multiple times.
        """
        self._registry._ensure_discovered()
