# scverse-backends

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/scverse-backends/main.svg)](https://results.pre-commit.ci/latest/github/scverse/scverse-backends/main)
[![codecov](https://codecov.io/gh/scverse/scverse-backends/graph/badge.svg?token=BMSUQWKIN8)](https://codecov.io/gh/scverse/scverse-backends)
[![Documentation Status](https://readthedocs.org/projects/scverse-backends/badge/?version=latest)](https://scverse-backends.readthedocs.io/en/latest/?badge=latest)

> ⚠️ **Under active development.** APIs may shift.

The default plugin & dispatch mechanism for [scverse](https://scverse.org).
Any host library decorates its public functions with `@backend_dispatch`; any
backend — GPU, distributed, JAX, PyTorch, anything
— plugs in via a Python entrypoint and gets picked up automatically.

Want to add a PyTorch backend, a JAX backend, your own custom one?
**You don't need a PR against the host.** Ship a package that exposes
a module or object with `name`, `aliases`, and host-named callables,
register it as an entry point, and users install it next to the host.
That's the whole contract.

```python
import example_host as eh

with eh.settings.use_backend("accelerated"):
    eh.some_function(data)
```

## Status

- Built from backend-dispatch prototypes in scverse host integrations.
- Concrete backend aliases belong in host integrations, not in this infrastructure package.
- Goal: host libraries depend on this package and stop re-rolling their own dispatch.

## Docs

Full docs at [scverse-backends.readthedocs.io](https://scverse-backends.readthedocs.io/en/latest/).

## License

MIT.
