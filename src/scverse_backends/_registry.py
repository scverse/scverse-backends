"""Backend discovery via Python entrypoints."""

from __future__ import annotations

import importlib.metadata
import logging
import re
import threading
import types
import warnings
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_RESERVED_BACKEND_NAMES = frozenset({"cpu"})
_BACKEND_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9_])?$")
_ALIAS_CONTAINER_TYPES = (list, tuple, set, frozenset)
_MISSING = object()


def _is_valid_backend_label(name: str) -> bool:
    """Return whether ``name`` is safe for backend names and aliases."""
    return bool(_BACKEND_LABEL_RE.fullmatch(name))


def _validate_config_label(
    name: str,
    *,
    description: str,
    canonical: str | None = None,
) -> None:
    """Validate host-owned backend labels eagerly."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{description} must be a non-empty string.")
    if not _is_valid_backend_label(name):
        owner = f" for {canonical!r}" if canonical is not None else ""
        raise ValueError(
            f"Invalid {description}{owner} {name!r}: use ASCII letters, "
            "numbers, dots, underscores, or hyphens; start with a letter or "
            "number; and do not end with a dot or hyphen."
        )
    if name in _RESERVED_BACKEND_NAMES:
        owner = f" for {canonical!r}" if canonical is not None else ""
        raise ValueError(f"Invalid {description}{owner}: {name!r} is reserved.")


def _trusted_aliases_from_config(canonical: str, aliases: Any) -> list[str]:
    """Validate and normalize trusted alias configuration."""
    if (
        aliases is None
        or isinstance(aliases, str)
        or not isinstance(aliases, _ALIAS_CONTAINER_TYPES)
    ):
        raise ValueError(
            f"Trusted backend aliases for {canonical!r} must be a list, tuple, "
            "or set of strings."
        )
    result: list[str] = []
    for alias in aliases:
        if not isinstance(alias, str):
            raise ValueError(
                f"Trusted backend aliases for {canonical!r} must contain only strings."
            )
        result.append(alias)
    return result


def _backend_aliases_from_instance(
    instance: Any,
    *,
    canonical: str,
    entrypoint_name: str,
) -> list[Any]:
    """Validate and normalize aliases exposed by a backend adapter."""
    aliases = getattr(instance, "aliases", _MISSING)
    if aliases is _MISSING:
        return []
    if aliases is None:
        warnings.warn(
            f"Ignoring aliases for backend {canonical!r} from entrypoint "
            f"{entrypoint_name!r}: backend.aliases must be a list, tuple, or "
            "set of strings, not None.",
            stacklevel=2,
        )
        return []
    if isinstance(aliases, str) or not isinstance(aliases, _ALIAS_CONTAINER_TYPES):
        warnings.warn(
            f"Ignoring aliases for backend {canonical!r} from entrypoint "
            f"{entrypoint_name!r}: backend.aliases must be a list, tuple, or "
            "set of strings.",
            stacklevel=2,
        )
        return []
    return list(aliases)


def _validate_requested_backend_name(name: str) -> None:
    """Validate user-provided backend selector type before fuzzy matching."""
    if not isinstance(name, str) or not name:
        raise ValueError("Backend name must be a non-empty string.")


def _normalize_distribution_name(name: str) -> str:
    """Normalize a Python distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _as_list(value: Any) -> list[str]:
    """Return ``value`` as a list of strings, accepting a scalar string."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _entrypoint_distribution_name(ep: importlib.metadata.EntryPoint) -> str | None:
    """Best-effort distribution name lookup for an entrypoint."""
    try:
        dist = ep.dist
        if dist is None:
            return None
        metadata = dist.metadata
    except Exception:  # noqa: BLE001
        return None
    try:
        return metadata["Name"]
    except KeyError:
        return None


def _coerce_backend_provider(provider: Any) -> Any:
    """Instantiate class entrypoints; use module/object entrypoints directly."""
    if isinstance(provider, type):
        return provider()
    return provider


def _provider_module_name(provider: Any) -> str:
    """Return the import module used for trust module-prefix checks."""
    if isinstance(provider, types.ModuleType):
        return provider.__name__
    return type(provider).__module__


def _module_matches_prefix(module: str, prefix: str) -> bool:
    """Return whether ``module`` is exactly or hierarchically under ``prefix``."""
    if prefix.endswith("."):
        return module.startswith(prefix)
    return module == prefix or module.startswith(f"{prefix}.")


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
        reserved_backends: dict[str, str] | None = None,
    ) -> None:
        self.entrypoint_group = entrypoint_group
        self.host_name = host_name
        self.trusted_backends: dict[str, dict[str, Any]] = dict(trusted_backends)
        self.reserved_backends: dict[str, str] = dict(reserved_backends or {})

        self._backends: dict[str, Any] = {}  # canonical_name -> instance
        self._alias_map: dict[str, str] = {}  # alias -> canonical_name
        self._load_errors: dict[str, BaseException] = {}
        self._registration_errors: dict[str, BaseException] = {}
        self._warned_untrusted: set[str] = set()
        self._discovered = False
        self._discovery_lock = threading.RLock()

        # Hook fired after the first successful discovery, so the dispatcher
        # can merge backend-specific params into already-decorated host
        # functions.
        self._on_discovered: Callable[[], None] | None = None

        # Build reverse lookup: alias -> canonical_name (for trusted backends)
        self._trusted_aliases: dict[str, str] = {}
        for reserved in self.reserved_backends:
            _validate_config_label(reserved, description="reserved backend name")
        for canonical, info in self.trusted_backends.items():
            _validate_config_label(canonical, description="trusted backend name")
            if canonical in self.reserved_backends:
                raise ValueError(
                    f"Trusted backend name {canonical!r} is reserved by "
                    f"{self.host_name}."
                )
            owner = self._trusted_aliases.get(canonical)
            if owner is not None and owner != canonical:
                raise ValueError(
                    f"Trusted backend name {canonical!r} is already configured "
                    f"as an alias for {owner!r}."
                )
            self._trusted_aliases[canonical] = canonical
            for alias in _trusted_aliases_from_config(
                canonical,
                info.get("aliases", []),
            ):
                _validate_config_label(
                    alias,
                    description="trusted backend alias",
                    canonical=canonical,
                )
                if alias in self.reserved_backends:
                    raise ValueError(
                        f"Trusted backend alias {alias!r} for {canonical!r} is "
                        f"reserved by {self.host_name}."
                    )
                owner = self._trusted_aliases.get(alias)
                if owner is not None and owner != canonical:
                    raise ValueError(
                        f"Trusted backend alias {alias!r} is configured for both "
                        f"{owner!r} and {canonical!r}."
                    )
                self._trusted_aliases[alias] = canonical

    def _ensure_discovered(self) -> None:
        """Discover and register backends via entrypoints (lazy, runs once)."""
        with self._discovery_lock:
            if self._discovered:
                return
            self._discovered = True

            for ep in importlib.metadata.entry_points(group=self.entrypoint_group):
                try:
                    provider = _coerce_backend_provider(ep.load())
                except Exception as e:  # noqa: BLE001
                    self._load_errors[ep.name] = e
                    logger.debug(
                        "Failed to load backend entrypoint %r", ep.name, exc_info=True
                    )
                else:
                    self._register_backend(
                        provider,
                        entrypoint_name=ep.name,
                        distribution_name=_entrypoint_distribution_name(ep),
                        object_ref=ep.value,
                    )

            if self._backends and self._on_discovered is not None:
                self._on_discovered()

    def _register_backend(
        self,
        instance: Any,
        *,
        entrypoint_name: str,
        distribution_name: str | None = None,
        object_ref: str | None = None,
    ) -> None:
        """Register a backend provider discovered from an entrypoint."""
        canonical = getattr(instance, "name", None)
        if not isinstance(canonical, str) or not canonical:
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: "
                "backend.name must be a non-empty string.",
                stacklevel=2,
            )
            return

        if not _is_valid_backend_label(canonical):
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: backend name "
                f"{canonical!r} is not a safe backend label. Use ASCII letters, "
                "numbers, dots, underscores, or hyphens; start with a letter or "
                "number; and do not end with a dot or hyphen.",
                stacklevel=2,
            )
            return

        if canonical in _RESERVED_BACKEND_NAMES:
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: "
                f"{canonical!r} is a reserved backend name.",
                stacklevel=2,
            )
            return

        if canonical in self.reserved_backends:
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: backend name "
                f"{canonical!r} is reserved by {self.host_name}. "
                f"{self.reserved_backends[canonical]}",
                stacklevel=2,
            )
            return

        if canonical in self.trusted_backends and not self._verify_trusted_provider(
            canonical,
            instance=instance,
            entrypoint_name=entrypoint_name,
            distribution_name=distribution_name,
            object_ref=object_ref,
        ):
            return

        trusted_owner = self._trusted_aliases.get(canonical)
        if trusted_owner is not None and trusted_owner != canonical:
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: backend name "
                f"{canonical!r} is reserved as an alias for trusted backend "
                f"{trusted_owner!r}.",
                stacklevel=2,
            )
            return

        existing = self._alias_map.get(canonical)
        if existing is not None and existing != canonical:
            warnings.warn(
                f"Ignoring backend entrypoint {entrypoint_name!r}: backend name "
                f"{canonical!r} is already used as an alias for {existing!r}.",
                stacklevel=2,
            )
            return

        if canonical in self._backends:
            warnings.warn(
                f"Backend {canonical!r} registered by multiple entrypoints. "
                f"Using the first one and ignoring {entrypoint_name!r}.",
                stacklevel=2,
            )
            return

        self._backends[canonical] = instance
        self._registration_errors.pop(canonical, None)
        self._alias_map[canonical] = canonical

        for alias in _backend_aliases_from_instance(
            instance,
            canonical=canonical,
            entrypoint_name=entrypoint_name,
        ):
            if not isinstance(alias, str) or not alias:
                warnings.warn(
                    f"Ignoring invalid alias {alias!r} for backend {canonical!r}.",
                    stacklevel=2,
                )
                continue
            if not _is_valid_backend_label(alias):
                warnings.warn(
                    f"Ignoring alias {alias!r} for backend {canonical!r}: "
                    "aliases must be safe backend labels.",
                    stacklevel=2,
                )
                continue
            if alias in _RESERVED_BACKEND_NAMES:
                warnings.warn(
                    f"Ignoring alias {alias!r} for backend {canonical!r}: "
                    "the name is reserved.",
                    stacklevel=2,
                )
                continue
            if alias in self.reserved_backends:
                warnings.warn(
                    f"Ignoring alias {alias!r} for backend {canonical!r}: "
                    f"the alias is reserved by {self.host_name}. "
                    f"{self.reserved_backends[alias]}",
                    stacklevel=2,
                )
                continue
            trusted_owner = self._trusted_aliases.get(alias)
            if trusted_owner is not None and trusted_owner != canonical:
                warnings.warn(
                    f"Ignoring alias {alias!r} for backend {canonical!r}: "
                    f"the alias is reserved for trusted backend {trusted_owner!r}.",
                    stacklevel=2,
                )
                continue
            if alias in self._alias_map and self._alias_map[alias] != canonical:
                warnings.warn(
                    f"Backend alias {alias!r} claimed by both "
                    f"{self._alias_map[alias]!r} and {canonical!r}. "
                    f"Using {self._alias_map[alias]!r}.",
                    stacklevel=2,
                )
            else:
                self._alias_map[alias] = canonical

    def _verify_trusted_provider(
        self,
        canonical: str,
        *,
        instance: Any,
        entrypoint_name: str,
        distribution_name: str | None,
        object_ref: str | None,
    ) -> bool:
        """Verify that a trusted backend is provided by the configured package."""
        info = self.trusted_backends[canonical]

        distributions = _as_list(
            info.get("distributions")
            or info.get("distribution")
            or info.get("packages")
            or info.get("package")
        )
        if distributions:
            expected = {_normalize_distribution_name(name) for name in distributions}
            actual = (
                _normalize_distribution_name(distribution_name)
                if distribution_name is not None
                else None
            )
            if actual not in expected:
                expected_msg = ", ".join(sorted(distributions))
                actual_msg = distribution_name or "<unknown>"
                return self._reject_trusted_provider(
                    canonical,
                    f"Refusing trusted backend {canonical!r} from entrypoint "
                    f"{entrypoint_name!r}: expected distribution {expected_msg}; "
                    f"got {actual_msg}.",
                )

        entrypoints = _as_list(info.get("entrypoints") or info.get("entrypoint"))
        if entrypoints and entrypoint_name not in entrypoints:
            return self._reject_trusted_provider(
                canonical,
                f"Refusing trusted backend {canonical!r} from entrypoint "
                f"{entrypoint_name!r}: expected one of {sorted(entrypoints)}.",
            )

        object_refs = _as_list(info.get("object_refs") or info.get("object_ref"))
        if object_refs and object_ref not in object_refs:
            return self._reject_trusted_provider(
                canonical,
                f"Refusing trusted backend {canonical!r} from entrypoint "
                f"{entrypoint_name!r}: expected object reference "
                f"one of {sorted(object_refs)}; got {object_ref!r}.",
            )

        module_prefixes = _as_list(
            info.get("module_prefixes") or info.get("module_prefix")
        )
        if module_prefixes:
            module = _provider_module_name(instance)
            if not any(
                _module_matches_prefix(module, prefix) for prefix in module_prefixes
            ):
                return self._reject_trusted_provider(
                    canonical,
                    f"Refusing trusted backend {canonical!r} from entrypoint "
                    f"{entrypoint_name!r}: expected module prefix "
                    f"one of {sorted(module_prefixes)}; got {module!r}.",
                )

        return True

    def _reject_trusted_provider(self, canonical: str, message: str) -> bool:
        error = ImportError(message)
        self._registration_errors[canonical] = error
        warnings.warn(message, stacklevel=3)
        return False

    def check_trusted(self, name: str) -> None:
        """Emit a one-time warning if the backend is not in the trusted list."""
        canonical = self._alias_map.get(name, name)
        if canonical in self._warned_untrusted:
            return
        if canonical in self.trusted_backends or canonical not in self._backends:
            return
        self._warned_untrusted.add(canonical)
        warnings.warn(
            f"Backend {canonical!r} is not in {self.host_name}'s trusted backends list. "
            f"It may not have passed the conformance test suite. "
            f"Trusted backends: {sorted(self.trusted_backends)}.",
            stacklevel=3,
        )

    def suggest(self, name: str) -> str:
        """Build an error message with 'did you mean' suggestions."""
        _validate_requested_backend_name(name)
        self._ensure_discovered()
        all_names = sorted(
            set(
                list(self._alias_map.keys())
                + list(self._trusted_aliases.keys())
                + list(self.reserved_backends.keys())
                + list(self._load_errors.keys())
                + list(self._registration_errors.keys())
            )
        )
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
        _validate_requested_backend_name(name)
        self._ensure_discovered()
        if name == "cpu":
            return "cpu"
        return (
            self._alias_map.get(name)
            or self._trusted_aliases.get(name)
            or (name if name in self._load_errors else None)
            or (name if name in self._registration_errors else None)
        )

    def get_backend(self, name: str) -> Any | None:
        """Get backend instance by name or alias. Returns None for ``"cpu"``."""
        _validate_requested_backend_name(name)
        self._ensure_discovered()
        if name == "cpu":
            return None
        canonical = self._alias_map.get(name) or self._trusted_aliases.get(name)
        if canonical is None:
            return None
        return self._backends.get(canonical)

    def require_backend(self, name: str) -> tuple[str, Any | None]:
        """Resolve and validate a backend name.

        Returns the canonical name and backend instance. For ``"cpu"``, the
        backend instance is ``None``.
        """
        _validate_requested_backend_name(name)
        if name in self.reserved_backends:
            raise ValueError(
                f"Backend name {name!r} is reserved by {self.host_name}. "
                f"{self.reserved_backends[name]}"
            )

        canonical = self.resolve_name(name)
        if canonical is None:
            raise ValueError(self.suggest(name))
        if canonical == "cpu":
            return canonical, None

        backend = self.get_backend(canonical)
        if backend is not None:
            self.check_trusted(canonical)
            return canonical, backend

        load_error = self._load_errors.get(canonical) or self._load_errors.get(name)
        if load_error is not None:
            raise ImportError(
                f"Backend {name!r} ({canonical}) failed to load from entrypoint "
                f"{canonical!r}."
            ) from load_error

        registration_error = self._registration_errors.get(
            canonical
        ) or self._registration_errors.get(name)
        if registration_error is not None:
            raise ImportError(str(registration_error)) from registration_error

        if self.is_trusted(canonical):
            package = self.trusted_package(canonical) or canonical
            raise ImportError(
                f"Backend {name!r} ({canonical}) is not installed. "
                f"Install it with: pip install {package}"
            )

        raise ImportError(f"Backend {name!r} is not installed.")

    def available_backend_names(self) -> list[str]:
        """Return all registered backend names and aliases."""
        self._ensure_discovered()
        return sorted(self._alias_map.keys())

    def is_trusted(self, canonical: str) -> bool:
        return canonical in self.trusted_backends

    def trusted_package(self, canonical: str) -> str | None:
        info = self.trusted_backends.get(canonical)
        return info.get("package") if info else None
