# Plugging in a backend

A *backend* is a Python package that provides alternative implementations
for one or more functions in a host library. Common examples are GPU,
distributed, JAX, or PyTorch implementations.

The backend contract is intentionally small: **no inheritance, no
`scverse-backends` import, no base class**. A backend exposes a module or
object with metadata and callables; the host discovers it through Python
entry points.

## Recommended layout

Use one private adapter module per host library:

```text
my_backend/
    _backends/
        __init__.py
        example_host.py
        another_host.py
```

This keeps host-integration glue in one predictable place and avoids
leaking adapter metadata (`name`, `aliases`) into a backend's public
analysis namespaces.

## Adapter modules

An adapter module needs three things:

1. `name` — the canonical backend name.
2. `aliases` — optional concrete names users can pass to
   `settings.backend = ...` or `backend=...`.
3. Callables named after the host functions the backend implements.

```python
# my_backend/_backends/example_host.py
from __future__ import annotations

from my_backend.example_host_impl import compute_score, embed, summarize

name = "my_backend"
aliases = ["mine", "cuda"]

__all__ = ["compute_score", "embed", "summarize"]
```

If the implementations live across several backend packages, gather them
in the adapter:

```python
# my_backend/_backends/another_host.py
from __future__ import annotations

from my_backend.array_ops import normalize, transform
from my_backend.models import fit, predict

name = "my_backend"
aliases = ["mine", "cuda"]

__all__ = [
    "normalize",
    "transform",
    "fit",
    "predict",
]
```

Backend names and aliases must be safe labels: ASCII letters, numbers,
dots, underscores, or hyphens; they must start with a letter or number,
and cannot end with a dot or hyphen. `cpu` is always reserved for the host
implementation. Hosts may reserve additional generic names such as `gpu`,
so prefer concrete aliases like `cuda`, `jax`, or a project abbreviation.

## Entry points

Register each adapter under the host's backend entry-point group:

```toml
# pyproject.toml
[project.entry-points."example_host.backends"]
my_backend = "my_backend._backends.example_host"

[project.entry-points."another_host.backends"]
my_backend = "my_backend._backends.another_host"
```

After users install the backend package next to the host, the host
discovers it lazily:

```python
import example_host as eh

eh.settings.backend = "cuda"
eh.compute_score(data, method="fast", batch_size=4096)
```

Module entry points are preferred. Class or object entry points are still
supported when a backend genuinely needs instance state:

```toml
[project.entry-points."example_host.backends"]
my_backend = "my_backend._backends.example_host:Backend"
```

## Lazy adapters

If importing the implementation package starts a heavy runtime, keep the
adapter itself cheap and load functions on first access:

```python
# my_backend/_backends/example_host.py
from __future__ import annotations

import importlib

name = "my_backend"
aliases = ["mine", "cuda"]

_functions = {
    "compute_score": "my_backend.example_host_impl",
    "embed": "my_backend.example_host_impl",
    "summarize": "my_backend.example_host_impl",
}

__all__ = ["compute_score", "embed", "summarize"]


def __getattr__(attr_name: str):
    if attr_name in _functions:
        func = getattr(importlib.import_module(_functions[attr_name]), attr_name)
        globals()[attr_name] = func
        return func
    raise AttributeError(f"{__name__!r} has no attribute {attr_name!r}")
```

This lets the host inspect the adapter metadata without importing CUDA,
JAX, or another heavy runtime during normal host import.

## Function signatures

Backend callables should use the same names as host functions for shared
parameters. Any extra public keyword-capable parameters are treated as
backend-only parameters.

```python
def compute_score(
    data,
    *,
    method="fast",
    batch_size=None,
    copy=False,
):
    """Compute a score on the backend.

    Parameters
    ----------
    data
        Input data.
    method
        Scoring method.
    batch_size
        Backend batch size.
    copy
        Return a copy.
    """
    ...
```

At dispatch time:

| parameter classification | dispatched call |
| --- | --- |
| in both host and backend signatures | forwarded to the backend |
| backend-only | forwarded to the backend |
| host-only at default value | silently dropped |
| host-only at non-default value | dropped with a warning |

After discovery, backend-only parameters are merged into the host
signature and docstring under `Other Parameters`. The provider is shown
compactly on the parameter line:

```text
Other Parameters
----------------
batch_size (my_backend)
    Backend batch size.
```

Private parameters such as `_internal` are not injected into host docs.
Use public keyword-only parameters for user-facing backend options.

## Trusted backends

Trust is owned by the host. A backend cannot make itself trusted by
setting a flag in its own adapter.

If a host lists your backend in `trusted_backends`, the host reserves your
canonical name and trusted aliases and can verify that the entry point was
provided by the expected distribution, entry-point name, object reference,
or module prefix. Backends not listed as trusted still work, but users see
a first-use warning.

To become trusted:

1. Add the backend to the host's `trusted_backends` configuration.
2. Consider running the host's reusable {doc}`feedback tests <conformance>`
   in the backend CI.
3. Document the backend alias users should choose.

Example host trust configuration:

```python
_dispatcher = BackendDispatcher(
    entrypoint_group="example_host.backends",
    host_name="example_host",
    trusted_backends={
        "my_backend": {
            "aliases": ["mine", "cuda"],
            "package": "my-backend",
            "entrypoints": ["my_backend"],
            "module_prefixes": ["my_backend"],
        },
    },
)
```

Matching backend entry point:

```toml
[project]
name = "my-backend"

[project.entry-points."example_host.backends"]
my_backend = "my_backend._backends.example_host"
```

## Backend checklist

- Keep one adapter module per host under a private namespace such as
  `_backends/`.
- Register module entry points in `pyproject.toml`.
- Use concrete aliases; do not claim `cpu` or host-reserved generic names.
- Keep adapter imports cheap, or use `__getattr__` for lazy loading.
- Match host function names and shared parameter names.
- Put backend-only option docs in the backend function's numpydoc.
- Consider running host feedback tests in backend CI.
