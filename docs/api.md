# API reference

## Public API

`BackendDispatcher` is the host-owned entry point. A host normally re-exports
these members:

| member | purpose |
| --- | --- |
| `backend_dispatch` | decorate a module-level function |
| `backend_class` | decorate a completely replaceable class |
| `settings` | select a default backend globally or in a context |
| `get_backend(name)` | retrieve a discovered adapter |
| `available_backend_names()` | list registered canonical names and aliases |

```{eval-rst}
.. currentmodule:: scverse_backends

.. autosummary::
   :toctree: generated
   :nosignatures:

   BackendDispatcher
   Settings
```

## Testing utilities

```{eval-rst}
.. currentmodule:: scverse_backends.testing

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_conformance
```
