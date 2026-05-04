# Wiring a host library

A *host* library is one that defines a public scientific API and wants
to let backends optionally accelerate it.

## 1. Instantiate a `BackendDispatcher`

```python
# example_host/_backends/__init__.py
from scverse_backends import BackendDispatcher

_dispatcher = BackendDispatcher(
    entrypoint_group="example_host.backends",
    host_name="example_host",
    trusted_backends={
        "example_accel": {
            "aliases": ["example", "accelerated"],
            "package": "example-host-accel",
        },
    },
    reserved_backends={
        "generic": "Use a concrete backend alias instead.",
    },
)

dispatch = _dispatcher.dispatch
settings = _dispatcher.settings
get_backend = _dispatcher.get_backend
available_backend_names = _dispatcher.available_backend_names
```

`BackendDispatcher` is fully **instance-scoped** — every host owns its
own registry, settings, and `ContextVar`. Multiple hosts can dispatch
independently in the same process.

### Constructor parameters

`entrypoint_group`
: The Python entrypoint group your backends register against. Convention
  is `"<host_name>.backends"`.

`host_name`
: Used in error messages and as the prefix of the settings `ContextVar`.

`trusted_backends` (optional)
: Mapping of canonical backend name to
  `{"aliases": [...], "package": "<pip-name>"}`. Backends listed here
  skip the "untrusted" warning on first use, and the host's error
  messages can suggest a `pip install` for known-but-not-installed
  backends. Any package can still plug in without being listed.

  Trusted aliases are reserved. If `accelerated` is listed as an alias for
  `example_accel`, an unrelated backend cannot claim
  `name="accelerated"` or `aliases=["accelerated"]`.

  Trusted canonical names are also provider-verified during entrypoint
  discovery. By default, `package` is treated as the expected Python
  distribution name. If the same trusted backend can be shipped by multiple
  distributions, use `distributions`:

  ```python
  trusted_backends={
      "example_accel": {
          "aliases": ["example", "accelerated"],
          "package": "example-host-accel",
          "distributions": [
              "example-host-accel",
              "example-host-accel-cu12",
          ],
          "entrypoints": ["example_accel"],
          "module_prefixes": ["example_host_accel"],
      },
  }
  ```

  This prevents another installed package from registering an entrypoint whose
  adapter claims `name="example_accel"` unless the entrypoint is provided
  by one of the configured distributions. `entrypoints`, `object_refs`, and
  `module_prefixes` are optional extra checks for hosts that want a narrower
  handshake.

`reserved_backends` (optional)
: Mapping of host-reserved backend names or aliases to human-readable reasons.
  No backend may claim these names. Use this for names that are too generic or
  intentionally left undefined by the host.

## 2. Decorate your public functions

Use `@dispatch` on module-level public functions. Instance methods and
class methods are outside the dispatch contract because their `self`/`cls`
binding does not map cleanly onto backend adapter callables.

```python
# example_host/analysis.py
from example_host._backends import dispatch

@dispatch
def compute_score(data, *, method="fast", n_jobs=None, copy=False):
    """Compute a score.

    Parameters
    ----------
    data
        Input data.
    method
        Scoring method.
    n_jobs
        Number of CPU jobs (host-only — silently dropped on backends).
    copy
        Return a copy instead of writing to ``adata``.
    """
    ...
```

`@dispatch` injects a `backend=None` keyword and (after backends are
discovered) merges any backend-specific parameters into the function's
signature and numpydoc. `None` means "use the active setting";
`backend="cpu"` explicitly forces the host implementation for that call.
The injected `backend` selector is documented with the host-owned
`Parameters`, while backend-specific parameters are placed under
`Other Parameters` and marked on the parameter line with the backend that
provided them. That
keeps host-owned knobs visually separate from optional backend package
knobs.

For example, a backend-only parameter from `example_accel` is rendered
as `solver (example_accel)` under `Other Parameters`.

## 3. Re-export `settings` for users

```python
# example_host/__init__.py
from example_host._backends import settings  # noqa: F401
```

## 4. Trigger eager discovery where signatures matter

Discovery is **lazy by default** — backend entrypoints aren't loaded
until something actually queries the dispatcher (settings setter,
`get_backend`, a non-CPU dispatched call). That keeps host import
fast.

But `help(my_func)`, IDE tooltips, and Sphinx autodoc introspect the
function statically — they never trigger lazy discovery. Without an
explicit nudge, those tools see only the host's CPU-only signature.

`BackendDispatcher.discover()` is the explicit nudge:

```python
# docs/conf.py
import example_host._backends as _b

_b._dispatcher.discover()  # so autodoc sees merged signatures
```

```python
# in IDE startup, test setup, or anywhere else that introspects:
from example_host._backends import _dispatcher
_dispatcher.discover()
```

`discover()` is idempotent and safe to call repeatedly — internally it
reuses the cached "original docstring" of every decorated function so
re-running the merge never double-injects backend params.

Discovery is intentionally one-shot for installed entry points. It is not
a runtime plugin reload mechanism; if a backend package is installed into
an already-running Python process, restart the process before expecting the
host to discover it. Functions decorated after discovery still get merged
against the already-discovered backends.

**Don't call `discover()` from your top-level `__init__.py`.** That
forces every backend's heavy imports (CUDA runtimes, JAX, …) to load on
every host import — wiping out the lazy-discovery benefit. Save it for
the few places that actually need merged signatures.

That's all the host needs to do. Backend authors take it from here.
