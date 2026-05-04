# scverse-backends

Pluggable backend dispatch machinery for [scverse](https://scverse.org) host
libraries (squidpy, scanpy, …). A *host* library decorates its public
functions with `@dispatch`; a *backend* library (e.g.
[rapids-singlecell](https://github.com/scverse/rapids_singlecell)) registers
itself via a Python entrypoint group, and decorated calls route to the
backend at runtime.

This package contains no host- or backend-specific logic — it is pure
dispatch machinery: entrypoint discovery, signature introspection, kwarg
routing, numpydoc merging, and a conformance test harness.

## Usage (host library)

```python
# squidpy/_backends/__init__.py
from scverse_backends import BackendDispatcher

_dispatcher = BackendDispatcher(
    entrypoint_group="squidpy.backends",
    host_name="squidpy",
    trusted_backends={
        "rapids_singlecell": {
            "aliases": ["rapids-singlecell", "rsc", "cuda", "gpu"],
            "package": "rapids-singlecell",
        },
    },
)

dispatch = _dispatcher.dispatch
settings = _dispatcher.settings
get_backend = _dispatcher.get_backend
available_backend_names = _dispatcher.available_backend_names
```

```python
# squidpy/gr/_ppatterns.py
from squidpy._backends import dispatch

@dispatch
def spatial_autocorr(adata, *, mode="moran", n_jobs=None, copy=False): ...
```

## Usage (backend library)

```toml
# rapids-singlecell/pyproject.toml
[project.entry-points."squidpy.backends"]
rapids_singlecell = "rapids_singlecell.squidpy_backend:RscSquidpyBackend"
```

```python
# rapids_singlecell/squidpy_backend.py
class RscSquidpyBackend:
    name = "rapids_singlecell"
    aliases = ["rapids-singlecell", "rsc", "cuda", "gpu"]

    def spatial_autocorr(self, adata, *, mode="moran", use_sparse=True, multi_gpu=None, copy=False):
        ...
```

## Calling a dispatched function

```python
import squidpy as sq

# Per-call backend
sq.gr.spatial_autocorr(adata, mode="moran", backend="gpu")

# Global
sq.settings.backend = "gpu"
sq.gr.spatial_autocorr(adata, mode="moran")

# Scoped
with sq.settings.use_backend("gpu"):
    sq.gr.co_occurrence(adata, cluster_key="cell_type")
```

## Argument routing

`@dispatch` introspects the signatures of both the host function and the
backend method:

| param classification | dispatched call |
| --- | --- |
| in both signatures (e.g. `adata`, `mode`, `copy`) | forwarded to backend |
| backend-only (e.g. `use_sparse`, `multi_gpu`) | forwarded; injected into host signature & docstring |
| host-only (e.g. `n_jobs`) at default value | silently dropped |
| host-only at non-default value | dropped with a warning |

## Conformance testing

`scverse_backends.testing.run_conformance` runs a host-supplied test suite
against a backend, comparing results to a CPU reference. Hosts ship the
test functions; backends call `run_conformance` from their own CI.

## Status

Early. Lifted from [scverse/squidpy#1151](https://github.com/scverse/squidpy/pull/1151).
The squidpy `_backends/` and rapids-singlecell `squidpy_backend.py` are the
proving grounds. Once stable, hosts will depend on this package directly.
