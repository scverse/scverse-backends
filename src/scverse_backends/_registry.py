"""Backend discovery via Python entrypoints."""

from __future__ import annotations

import importlib.metadata
import logging
import warnings
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _Registry:
    """Per-host backend registry.

    Holds the entrypoint group to scan, the trusted-backends list, and the
    discovered backend instances. All host-scoped state lives here so multiple
    host libraries can coexist in the same process.
    """

    def __init__(
        self,
        *,
        entrypoint_group: str,
        host_name: str,
        trusted_backends: dict[str, dict[str, Any]],
    ) -> None:
        self.entrypoint_group = entrypoint_group
        self.host_name = host_name
        self.trusted_backends: dict[str, dict[str, Any]] = dict(trusted_backends)

        self._backends: dict[str, Any] = {}  # canonical_name -> instance
        self._alias_map: dict[str, str] = {}  # alias -> canonical_name
        self._discovered = False

        # Hook fired after the first successful discovery, so the dispatcher
        # can merge backend-specific params into already-decorated host
        # functions.
        self._on_discovered: Callable[[], None] | None = None

        # Build reverse lookup: alias -> canonical_name (for trusted backends)
        self._trusted_aliases: dict[str, str] = {}
        for canonical, info in self.trusted_backends.items():
            self._trusted_aliases[canonical] = canonical
            for alias in info.get("aliases", []):
                self._trusted_aliases[alias] = canonical

    def _ensure_discovered(self) -> None:
        """Discover and register backends via entrypoints (lazy, runs once)."""
        if self._discovered:
            return
        self._discovered = True

        for ep in importlib.metadata.entry_points(group=self.entrypoint_group):
            try:
                cls = ep.load()
                instance = cls()
                canonical = instance.name
                self._backends[canonical] = instance
                self._alias_map[canonical] = canonical
                for alias in getattr(instance, "aliases", []):
                    if alias in self._alias_map and self._alias_map[alias] != canonical:
                        warnings.warn(
                            f"Backend alias {alias!r} claimed by both "
                            f"{self._alias_map[alias]!r} and {canonical!r}. "
                            f"Using {self._alias_map[alias]!r}.",
                            stacklevel=2,
                        )
                    else:
                        self._alias_map[alias] = canonical
            except Exception:  # noqa: BLE001
                logger.debug("Failed to load backend entrypoint %r", ep.name, exc_info=True)

        if self._backends and self._on_discovered is not None:
            self._on_discovered()

    def check_trusted(self, name: str) -> None:
        """Emit a one-time warning if the backend is not in the trusted list."""
        canonical = self._alias_map.get(name, name)
        if canonical not in self.trusted_backends and canonical in self._backends:
            warnings.warn(
                f"Backend {canonical!r} is not in {self.host_name}'s trusted backends list. "
                f"It may not have passed the conformance test suite. "
                f"Trusted backends: {sorted(self.trusted_backends)}.",
                stacklevel=3,
            )

    def suggest(self, name: str) -> str:
        """Build an error message with 'did you mean' suggestions."""
        self._ensure_discovered()
        all_names = sorted(set(list(self._alias_map.keys()) + list(self._trusted_aliases.keys())))
        matches = get_close_matches(name, all_names, n=1, cutoff=0.4)
        msg = f"Unknown backend {name!r}."
        if matches:
            msg += f" Did you mean {matches[0]!r}?"
        available = self.available_backend_names()
        if available:
            msg += f" Available: {available}."
        else:
            msg += " No backends are currently installed."
        return msg

    def resolve_name(self, name: str) -> str | None:
        """Resolve alias to canonical backend name.

        Recognises both loaded backends and trusted (but not installed) aliases.
        Returns ``None`` only for completely unknown names.
        """
        self._ensure_discovered()
        if name == "cpu":
            return "cpu"
        return self._alias_map.get(name) or self._trusted_aliases.get(name)

    def get_backend(self, name: str) -> Any | None:
        """Get backend instance by name or alias. Returns None for ``"cpu"``."""
        self._ensure_discovered()
        if name == "cpu":
            return None
        canonical = self._alias_map.get(name) or self._trusted_aliases.get(name)
        if canonical is None:
            return None
        return self._backends.get(canonical)

    def available_backend_names(self) -> list[str]:
        """Return all registered backend names and aliases."""
        self._ensure_discovered()
        return sorted(self._alias_map.keys())

    def is_trusted(self, canonical: str) -> bool:
        return canonical in self.trusted_backends

    def trusted_package(self, canonical: str) -> str | None:
        info = self.trusted_backends.get(canonical)
        return info.get("package") if info else None
