"""Dispatch decorator with introspection-based argument routing."""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from scverse_backends._registry import _Registry
    from scverse_backends._settings import _Settings


# numpydoc section headers that end a Parameters block
_NUMPYDOC_SECTIONS = frozenset(
    ("Returns", "Raises", "See Also", "Notes", "Examples", "Yields", "Warns", "References", "Attributes", "Methods")
)


def _find_section(lines: list[str], section: str) -> tuple[int, int] | None:
    """Find a numpydoc section, returning ``(header_line, first_content_line)``."""
    for i, line in enumerate(lines):
        if line.strip() == section and i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
            return i, i + 2
    return None


def _detect_indent(lines: list[str], start: int, end: int) -> str:
    """Detect the parameter-name indentation used in a numpydoc Parameters block."""
    for line in lines[start:end]:
        stripped = line.lstrip()
        if stripped and stripped.split()[0].replace("*", "").replace(",", "").isidentifier():
            return line[: len(line) - len(stripped)]
    return "    "


def _extract_param_docs(docstring: str | None, param_names: set[str]) -> dict[str, str]:
    """Extract numpydoc parameter entries for the given names.

    Uses indentation-based parsing: a parameter entry starts with a line
    whose indentation matches the section's base indent, and continues
    with all subsequent lines that are blank or more deeply indented.

    Returns a dict mapping param name to its dedented doc block.
    On any parse ambiguity, the parameter is skipped rather than producing
    garbled output.
    """
    if not docstring or not param_names:
        return {}

    lines = docstring.split("\n")
    section = _find_section(lines, "Parameters")
    if section is None:
        return {}

    _, content_start = section

    # Find where the Parameters section ends
    content_end = len(lines)
    for i in range(content_start, len(lines)):
        stripped = lines[i].strip()
        if stripped in _NUMPYDOC_SECTIONS:
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                content_end = i
                break

    base_indent = _detect_indent(lines, content_start, content_end)
    base_indent_len = len(base_indent)

    result: dict[str, str] = {}
    i = content_start
    while i < content_end:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        line_indent_len = len(line) - len(line.lstrip())
        if line_indent_len != base_indent_len:
            i += 1
            continue

        first_token = line.strip().split()[0].rstrip(",")
        name = first_token.lstrip("*")

        block_lines = [line]
        j = i + 1
        while j < content_end:
            body_line = lines[j]
            if not body_line.strip():
                block_lines.append(body_line)
                j += 1
                continue
            body_indent_len = len(body_line) - len(body_line.lstrip())
            if body_indent_len > base_indent_len:
                block_lines.append(body_line)
                j += 1
            else:
                break

        while block_lines and not block_lines[-1].strip():
            block_lines.pop()

        if name in param_names:
            dedented = []
            for bl in block_lines:
                if bl.strip():
                    dedented.append(bl[base_indent_len:] if len(bl) >= base_indent_len else bl)
                else:
                    dedented.append("")
            result[name] = "\n".join(dedented)

        i = j

    return result


def _inject_param_docs(docstring: str | None, extra_docs: dict[str, str]) -> str:
    """Inject extra parameter docs and ``backend`` doc into a numpydoc docstring.

    Inserts before the first non-Parameters section (Returns, Raises, etc.).
    If the Parameters section can't be found, the docstring is returned unchanged
    rather than producing garbled output.
    """
    if not docstring:
        return docstring or ""

    lines = docstring.split("\n")
    section = _find_section(lines, "Parameters")
    if section is None:
        return docstring

    _, content_start = section

    insert_idx = len(lines)
    for i in range(content_start, len(lines)):
        stripped = lines[i].strip()
        if stripped in _NUMPYDOC_SECTIONS:
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                insert_idx = i
                break

    indent = _detect_indent(lines, content_start, insert_idx)
    body_indent = indent + "    "

    extra_lines: list[str] = []
    for doc_block in extra_docs.values():
        for doc_line in doc_block.split("\n"):
            if doc_line.strip():
                extra_lines.append(indent + doc_line)
            else:
                extra_lines.append("")

    extra_lines.append(f"{indent}backend")
    extra_lines.append(f"{body_indent}Backend to use. Use ``'cpu'`` for the default implementation or a")
    extra_lines.append(f"{body_indent}registered backend name (e.g. ``'gpu'``).")

    lines = lines[:insert_idx] + extra_lines + [""] + lines[insert_idx:]
    return "\n".join(lines)


def _build_signature(func: Callable) -> None:
    """Build the wrapper's ``__signature__`` from the host function, adding ``backend``."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if "backend" not in sig.parameters:
        backend_param = inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default="cpu", annotation=str)
        kwargs_idx = next((i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_KEYWORD), None)
        if kwargs_idx is not None:
            params.insert(kwargs_idx, backend_param)
        else:
            params.append(backend_param)

    func.__signature__ = sig.replace(parameters=params)


def _find_public_func(wrapper: Callable) -> Callable:
    """Find the outermost public function that wraps a dispatch wrapper.

    Walks from the module-level attribute through ``__wrapped__`` to verify
    it actually chains back to our wrapper.  Returns the outermost function
    (which may be ``wrapper`` itself if no outer decorator exists).
    """
    import sys

    func = wrapper.__wrapped__
    mod = sys.modules.get(func.__module__)
    if mod is None:
        return wrapper

    candidate = getattr(mod, func.__name__, None)
    if candidate is None or candidate is wrapper:
        return wrapper

    obj = candidate
    while obj is not None:
        if obj is wrapper:
            return candidate
        obj = getattr(obj, "__wrapped__", None)

    return wrapper


class _Dispatch:
    """Per-host dispatch decorator + signature merging.

    Owns the list of decorated host functions and the ``(func, backend)``
    parameter-set cache, both scoped to a single ``BackendDispatcher``.
    """

    def __init__(self, registry: _Registry, settings: _Settings) -> None:
        self._registry = registry
        self._settings = settings
        # cache: (func_qualname, backend_canonical_name) -> (shared, host_only, backend_only, host_defaults)
        self._sig_cache: dict[tuple[str, str], tuple[set, set, set, dict]] = {}
        self._dispatched_functions: list[Callable] = []
        # Wire the post-discovery hook so backend-specific params get merged
        # into already-decorated host functions on first discovery.
        registry._on_discovered = self._update_signatures

    def _get_param_sets(
        self,
        func: Callable,
        adapter_method: Callable,
        backend_name: str,
    ) -> tuple[set, set, set, dict]:
        """Compute shared/host-only/backend-only param sets. Cached per (func, backend)."""
        key = (func.__qualname__, backend_name)
        if key in self._sig_cache:
            return self._sig_cache[key]

        host_sig = inspect.signature(func)
        adapter_sig = inspect.signature(adapter_method)

        host_params = set(host_sig.parameters.keys()) - {"self", "args", "kwargs"}
        adapter_params = set(adapter_sig.parameters.keys()) - {"self", "args", "kwargs"}

        # Remove "backend" — it's the dispatch kwarg, not forwarded
        host_params.discard("backend")
        adapter_params.discard("backend")

        shared = host_params & adapter_params
        host_only = host_params - adapter_params
        backend_only = adapter_params - host_params

        host_defaults: dict[str, Any] = {}
        for name, param in host_sig.parameters.items():
            if name in host_only and param.default is not inspect.Parameter.empty:
                host_defaults[name] = param.default

        result = (shared, host_only, backend_only, host_defaults)
        self._sig_cache[key] = result
        return result

    def _update_signatures(self) -> None:
        """Merge backend-only params from discovered backends into dispatched signatures."""
        for wrapper in self._dispatched_functions:
            func = wrapper.__wrapped__
            func_name = func.__name__
            host_sig = inspect.signature(func)
            host_param_names = set(host_sig.parameters.keys())
            host_param_names.add("backend")

            backend_params: list[inspect.Parameter] = []
            backend_param_names: set[str] = set()
            adapter_docs: dict[str, str] = {}
            for backend in self._registry._backends.values():
                try:
                    method = getattr(backend, func_name, None)
                except Exception:  # noqa: BLE001
                    continue
                if method is None:
                    continue
                try:
                    adapter_sig = inspect.signature(method)
                except (ValueError, TypeError):
                    continue

                new_names: set[str] = set()
                for name, param in adapter_sig.parameters.items():
                    if name in host_param_names or name in backend_param_names or name in {"self", "args", "kwargs"}:
                        continue
                    backend_params.append(param.replace(kind=inspect.Parameter.KEYWORD_ONLY))
                    backend_param_names.add(name)
                    new_names.add(name)

                if new_names:
                    param_docs = _extract_param_docs(method.__doc__, new_names)
                    adapter_docs.update(param_docs)

            params: list[inspect.Parameter] = []
            var_kw = None
            for p in host_sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    var_kw = p
                else:
                    params.append(p)

            params.extend(backend_params)
            params.append(inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default="cpu", annotation=str))
            if var_kw is not None:
                params.append(var_kw)

            merged_sig = host_sig.replace(parameters=params)
            merged_doc = _inject_param_docs(wrapper.__doc__, adapter_docs)

            wrapper.__signature__ = merged_sig
            wrapper.__doc__ = merged_doc

            public_func = _find_public_func(wrapper)
            if public_func is not wrapper:
                public_func.__signature__ = merged_sig
                public_func.__doc__ = merged_doc

    def decorator(self, func: Callable) -> Callable:
        """Route a function call to the active backend or fall back to CPU.

        Apply this decorator to any host public function that a backend may
        accelerate. The decorator:

        * Injects a ``backend`` keyword argument (default ``"cpu"``).
        * Injects backend-specific parameters and their docstrings from
          discovered backends into the function signature and docstring.
        * At call time, resolves the effective backend (``backend`` kwarg >
          ``settings.backend``) and forwards arguments via signature
          introspection.

        Argument routing (backend path):

        * **shared** (in both host and backend) — forwarded.
        * **backend-only** (e.g. ``use_sparse``, ``multi_gpu``) — forwarded.
        * **host-only at default value** — silently dropped.
        * **host-only at non-default value** — dropped with a warning.

        If the active backend does not implement the decorated function, the
        call falls back to the CPU implementation transparently.
        """
        func_name = func.__name__
        registry = self._registry
        settings = self._settings

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            local_backend = kwargs.pop("backend", None)
            effective = local_backend or settings.backend

            if effective == "cpu":
                return func(*args, **kwargs)

            backend = registry.get_backend(effective)
            if backend is None:
                raise RuntimeError(
                    f"Backend {effective!r} is not installed. "
                    f"Install it or set {registry.host_name}.settings.backend = 'cpu'."
                )

            method = getattr(backend, func_name, None)
            if method is None:
                # Backend doesn't implement this function — fall back to CPU
                return func(*args, **kwargs)

            shared, host_only, backend_only, host_defaults = self._get_param_sets(func, method, backend.name)

            adapter_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                if key in shared or key in backend_only:
                    adapter_kwargs[key] = value
                elif key in host_only:
                    if key not in host_defaults or value != host_defaults[key]:
                        warnings.warn(
                            f"{key!r} has no effect on backend {effective!r}.",
                            stacklevel=2,
                        )
                else:
                    # Unknown kwarg — let the adapter deal with it
                    adapter_kwargs[key] = value

            return method(*args, **adapter_kwargs)

        _build_signature(wrapper)
        self._dispatched_functions.append(wrapper)
        return wrapper
