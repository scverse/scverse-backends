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
    (
        "Returns",
        "Other Parameters",
        "Raises",
        "See Also",
        "Notes",
        "Examples",
        "Yields",
        "Warns",
        "References",
        "Attributes",
        "Methods",
    )
)
_RESERVED_BACKEND_PARAM_NAMES = frozenset({"self", "args", "kwargs", "backend"})


def _is_injectable_backend_param(name: str, param: inspect.Parameter) -> bool:
    """Return whether a backend-only param should become part of host docs."""
    if name in _RESERVED_BACKEND_PARAM_NAMES:
        return False
    if not name.isidentifier() or name.startswith("_"):
        return False
    return param.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _format_backend_source_suffix(sources: set[str] | None) -> str:
    """Format backend provider names for a compact parameter-line suffix."""
    if not sources:
        return ""
    return ", ".join(sorted(sources))


def _append_backend_source_suffix(header: str, suffix: str) -> str:
    """Append backend provenance to a numpydoc parameter header."""
    if not suffix:
        return header
    if ":" in header:
        name, rest = header.split(":", 1)
        return f"{name.rstrip()} ({suffix}) :{rest}"
    return f"{header.rstrip()} ({suffix})"


def _annotate_backend_doc(
    name: str,
    doc_block: str | None,
    sources: set[str] | None,
) -> str:
    """Add provenance to an injected backend parameter doc block."""
    lines = doc_block.split("\n") if doc_block else [name]
    if not lines or not lines[0].strip():
        lines = [name]
    lines[0] = _append_backend_source_suffix(
        lines[0],
        _format_backend_source_suffix(sources),
    )
    return "\n".join(lines)


def _routable_param_names(sig: inspect.Signature) -> set[str]:
    """Parameter names that can participate in name-based dispatch routing."""
    return {
        name
        for name, param in sig.parameters.items()
        if name not in _RESERVED_BACKEND_PARAM_NAMES
        and param.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def _callable_name(func: Callable) -> str:
    """Best-effort function name for dynamic callable objects."""
    return getattr(func, "__name__", type(func).__name__)


def _callable_qualname(func: Callable) -> str:
    """Best-effort qualified function name for dynamic callable objects."""
    return getattr(func, "__qualname__", _callable_name(func))


def _callable_module(func: Callable) -> str:
    """Best-effort module name for dynamic callable objects."""
    return getattr(func, "__module__", type(func).__module__)


def _find_section(lines: list[str], section: str) -> tuple[int, int] | None:
    """Find a numpydoc section, returning ``(header_line, first_content_line)``."""
    for i, line in enumerate(lines):
        if (
            line.strip() == section
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("---")
        ):
            return i, i + 2
    return None


def _find_section_end(lines: list[str], start: int) -> int:
    """Find the line where a numpydoc section body ends."""
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped in _NUMPYDOC_SECTIONS:
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                return i
    return len(lines)


def _splice_section_lines(
    lines: list[str],
    insert_idx: int,
    inserted: list[str],
) -> list[str]:
    """Insert section lines with one blank line around the inserted block."""
    prefix = lines[:insert_idx]
    suffix = lines[insert_idx:]
    while prefix and not prefix[-1].strip():
        prefix.pop()
    while suffix and not suffix[0].strip():
        suffix.pop(0)
    return prefix + ["", *inserted, ""] + suffix


def _detect_indent(lines: list[str], start: int, end: int) -> str:
    """Detect the parameter-name indentation used in a numpydoc Parameters block."""
    for line in lines[start:end]:
        stripped = line.lstrip()
        if (
            stripped
            and stripped.split()[0].replace("*", "").replace(",", "").isidentifier()
        ):
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
                    dedented.append(
                        bl[base_indent_len:] if len(bl) >= base_indent_len else bl
                    )
                else:
                    dedented.append("")
            result[name] = "\n".join(dedented)

        i = j

    return result


def _build_injected_param_doc_lines(
    *,
    indent: str,
    body_indent: str,
    extra_docs: dict[str, str],
    param_sources: dict[str, set[str]] | None,
) -> list[str]:
    """Build generated doc lines for injected backend parameters."""
    extra_lines: list[str] = []
    for name, doc_block in extra_docs.items():
        annotated = _annotate_backend_doc(
            name,
            doc_block,
            param_sources.get(name) if param_sources is not None else None,
        )
        for doc_line in annotated.split("\n"):
            if doc_line.strip():
                extra_lines.append(indent + doc_line)
            else:
                extra_lines.append("")

    return extra_lines


def _build_backend_doc_lines(*, indent: str, body_indent: str) -> list[str]:
    """Build generated doc lines for the host-level backend selector."""
    return [
        f"{indent}backend",
        f"{body_indent}Backend selector injected by ``scverse-backends``.",
        (
            f"{body_indent}If ``None``, use the active backend in settings. "
            "Use ``'cpu'``"
        ),
        f"{body_indent}for the host implementation.",
        f"{body_indent}Otherwise pass a registered backend name or alias.",
    ]


def _inject_param_docs(
    docstring: str | None,
    extra_docs: dict[str, str],
    param_sources: dict[str, set[str]] | None = None,
) -> str:
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
    parameters_end = _find_section_end(lines, content_start)
    indent = _detect_indent(lines, content_start, parameters_end)
    body_indent = indent + "    "
    backend_lines = _build_backend_doc_lines(indent=indent, body_indent=body_indent)
    lines = _splice_section_lines(lines, parameters_end, backend_lines)

    if not extra_docs:
        return "\n".join(lines)

    other_section = _find_section(lines, "Other Parameters")
    if other_section is not None:
        _, other_content_start = other_section
        other_end = _find_section_end(lines, other_content_start)
        indent = _detect_indent(lines, other_content_start, other_end)
        body_indent = indent + "    "
        extra_lines = _build_injected_param_doc_lines(
            indent=indent,
            body_indent=body_indent,
            extra_docs=extra_docs,
            param_sources=param_sources,
        )
        return "\n".join(_splice_section_lines(lines, other_end, extra_lines))

    section = _find_section(lines, "Parameters")
    if section is None:
        return "\n".join(lines)
    _, content_start = section
    parameters_end = _find_section_end(lines, content_start)

    extra_lines = _build_injected_param_doc_lines(
        indent=indent,
        body_indent=body_indent,
        extra_docs=extra_docs,
        param_sources=param_sources,
    )
    other_param_lines = ["Other Parameters", "----------------", *extra_lines]
    return "\n".join(_splice_section_lines(lines, parameters_end, other_param_lines))


def _build_signature(func: Callable) -> None:
    """Build the wrapper's ``__signature__`` from the host function, adding ``backend``."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if "backend" not in sig.parameters:
        backend_param = inspect.Parameter(
            "backend",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str | None,
        )
        kwargs_idx = next(
            (
                i
                for i, p in enumerate(params)
                if p.kind == inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )
        if kwargs_idx is not None:
            params.insert(kwargs_idx, backend_param)
        else:
            params.append(backend_param)

    setattr(func, "__signature__", sig.replace(parameters=params))


def _find_public_func(wrapper: Callable) -> Callable:
    """Find the outermost public function that wraps a dispatch wrapper.

    Walks from the module-level attribute through ``__wrapped__`` to verify
    it actually chains back to our wrapper.  Returns the outermost function
    (which may be ``wrapper`` itself if no outer decorator exists).
    """
    import sys

    func = getattr(wrapper, "__wrapped__", None)
    if func is None:
        return wrapper
    mod = sys.modules.get(_callable_module(func))
    if mod is None:
        return wrapper

    candidate = getattr(mod, _callable_name(func), None)
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
        # cache: (func_module, func_qualname, backend_canonical_name) -> param sets
        self._sig_cache: dict[tuple[str, str, str], tuple[set, set, set, dict]] = {}
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
        key = (_callable_module(func), _callable_qualname(func), backend_name)
        if key in self._sig_cache:
            return self._sig_cache[key]

        host_sig = inspect.signature(func)
        adapter_sig = inspect.signature(adapter_method)

        host_params = _routable_param_names(host_sig)
        adapter_params = _routable_param_names(adapter_sig)

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
        """Merge backend-only params from discovered backends into dispatched signatures.

        Idempotent: each wrapper's pre-merge ``__doc__`` is captured the first
        time signatures are updated, so re-running the merge never double-injects.
        """
        for wrapper in self._dispatched_functions:
            func = getattr(wrapper, "__wrapped__", None)
            if func is None:
                continue
            func_name = _callable_name(func)
            host_sig = inspect.signature(func)
            host_param_names = set(host_sig.parameters.keys())
            host_param_names.add("backend")

            backend_params: list[inspect.Parameter] = []
            backend_param_names: set[str] = set()
            adapter_docs: dict[str, str] = {}
            param_sources: dict[str, set[str]] = {}
            for backend_name, backend in self._registry._backends.items():
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

                candidate_names: set[str] = set()
                for name, param in adapter_sig.parameters.items():
                    if name in host_param_names or name in {"self", "args", "kwargs"}:
                        continue
                    if not _is_injectable_backend_param(name, param):
                        if param.kind not in {
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                            inspect.Parameter.POSITIONAL_ONLY,
                        }:
                            warnings.warn(
                                f"Ignoring backend-only parameter {name!r} from "
                                f"backend {backend_name!r} for {func_name!r}: "
                                "backend-only parameters must be public "
                                "keyword-capable Python identifiers.",
                                stacklevel=2,
                            )
                        continue
                    candidate_names.add(name)
                    param_sources.setdefault(name, set()).add(backend_name)
                    if name in backend_param_names:
                        continue
                    backend_params.append(
                        param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
                    )
                    backend_param_names.add(name)

                if candidate_names:
                    param_docs = _extract_param_docs(method.__doc__, candidate_names)
                    for name, doc in param_docs.items():
                        adapter_docs.setdefault(name, doc)

            for name in backend_param_names:
                adapter_docs.setdefault(name, name)

            params: list[inspect.Parameter] = []
            var_kw = None
            for p in host_sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    var_kw = p
                else:
                    params.append(p)

            params.extend(backend_params)
            params.append(
                inspect.Parameter(
                    "backend",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str | None,
                )
            )
            if var_kw is not None:
                params.append(var_kw)

            merged_sig = host_sig.replace(parameters=params)
            # Always merge from the pre-merge docstring so re-running discover()
            # doesn't stack multiple copies of backend params on top of each
            # other. Capturing lazily preserves doc mutations from outer
            # decorators applied after @dispatch.
            if not hasattr(wrapper, "__scverse_backends_base_doc__"):
                setattr(wrapper, "__scverse_backends_base_doc__", wrapper.__doc__)
            original_doc = getattr(wrapper, "__scverse_backends_base_doc__")
            merged_doc = _inject_param_docs(original_doc, adapter_docs, param_sources)

            setattr(wrapper, "__signature__", merged_sig)
            wrapper.__doc__ = merged_doc

            public_func = _find_public_func(wrapper)
            if public_func is not wrapper:
                setattr(public_func, "__signature__", merged_sig)
                public_func.__doc__ = merged_doc

    def decorator(self, func: Callable) -> Callable:
        """Route a function call to the active backend or fall back to CPU.

        Apply this decorator to any host public function that a backend may
        accelerate. The decorator:

        * Injects a ``backend`` keyword argument (default ``None``).
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
        if "backend" in inspect.signature(func).parameters:
            raise TypeError(
                f"Cannot dispatch {_callable_module(func)}.{_callable_qualname(func)}: "
                "'backend' is reserved for scverse-backends."
            )

        func_name = _callable_name(func)
        registry = self._registry
        settings = self._settings

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            local_backend = kwargs.pop("backend", None)
            effective = settings.backend if local_backend is None else local_backend

            if effective == "cpu":
                return func(*args, **kwargs)

            canonical, backend = registry.require_backend(effective)

            method = getattr(backend, func_name, None)
            if method is None:
                # Backend doesn't implement this function — fall back to CPU
                return func(*args, **kwargs)

            shared, host_only, backend_only, host_defaults = self._get_param_sets(
                func, method, canonical
            )

            adapter_kwargs = self._route_arguments(
                func=func,
                args=args,
                kwargs=kwargs,
                shared=shared,
                host_only=host_only,
                backend_only=backend_only,
                host_defaults=host_defaults,
                backend_name=effective,
            )
            return self._call_backend_method(method, adapter_kwargs)

        _build_signature(wrapper)
        self._dispatched_functions.append(wrapper)
        return wrapper

    @staticmethod
    def _route_arguments(
        *,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        shared: set[str],
        host_only: set[str],
        backend_only: set[str],
        host_defaults: dict[str, Any],
        backend_name: str,
    ) -> dict[str, Any]:
        """Bind against the host signature, then build backend kwargs by name."""
        host_sig = inspect.signature(func)

        host_kwargs: dict[str, Any] = {}
        adapter_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in backend_only:
                adapter_kwargs[key] = value
            elif key in host_sig.parameters:
                host_kwargs[key] = value
            else:
                # Unknown kwarg — let the adapter raise or consume it.
                adapter_kwargs[key] = value

        bound = host_sig.bind(*args, **host_kwargs)
        bound.apply_defaults()

        for key, value in bound.arguments.items():
            param = host_sig.parameters[key]
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if key in shared:
                adapter_kwargs[key] = value
            elif key in host_only:
                default = host_defaults.get(key, inspect.Parameter.empty)
                if default is inspect.Parameter.empty or value != default:
                    warnings.warn(
                        f"{key!r} has no effect on backend {backend_name!r}.",
                        stacklevel=3,
                    )

        return adapter_kwargs

    @staticmethod
    def _call_backend_method(method: Callable, adapter_kwargs: dict[str, Any]) -> Any:
        """Call a backend method, preserving positional-only adapter params."""
        adapter_sig = inspect.signature(method)
        adapter_args: list[Any] = []
        call_kwargs: dict[str, Any] = {}
        consumed: set[str] = set()

        for name, param in adapter_sig.parameters.items():
            if name == "self" or param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if name not in adapter_kwargs:
                continue

            consumed.add(name)
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                adapter_args.append(adapter_kwargs[name])
            else:
                call_kwargs[name] = adapter_kwargs[name]

        for name, value in adapter_kwargs.items():
            if name not in consumed:
                call_kwargs[name] = value

        return method(*adapter_args, **call_kwargs)
