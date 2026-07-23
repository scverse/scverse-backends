# scverse-backends

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/scverse-backends/main.svg)](https://results.pre-commit.ci/latest/github/scverse/scverse-backends/main)
[![codecov](https://codecov.io/gh/scverse/scverse-backends/graph/badge.svg?token=BMSUQWKIN8)](https://codecov.io/gh/scverse/scverse-backends)
[![Documentation Status](https://readthedocs.org/projects/scverse-backends/badge/?version=latest)](https://scverse-backends.readthedocs.io/en/latest/?badge=latest)

```{warning}
This repository is under active development. APIs may shift before the
first stable release.
```

**The default plugin and dispatch mechanism for
[scverse](https://scverse.org).** Host libraries mark public functions with
`@backend_dispatch` and replaceable classes with `@backend_class`. GPU,
distributed, JAX, PyTorch, and other backends plug in through Python entry
points and are discovered automatically.

## At a glance

```python
import example_host as eh

# Per-call backend
eh.compute_score(data, method="fast", backend="cuda")

# Complete class replacement
model = eh.Neighborhood(data, backend="cuda")

# Current context
eh.settings.backend = "cuda"
eh.compute_score(data, method="fast")

# Scoped
with eh.settings.use_backend("cuda"):
    eh.embed(data, n_components=32)
```

## Want to add a backend?

**No PR against the host needed.** Ship a package that exposes a module or
object with `name`, `aliases`, and functions or classes named after the host
APIs you implement. Register it under the host's entrypoint group, and users
install it next to the host. See {doc}`usage/backend`.

This is the path for a PyTorch backend, a JAX backend, a Dask backend,
or anything else. The host library doesn't need to know you exist.

## Why this exists

Every scverse host that wants a GPU/distributed/alt backend was
re-implementing the same dispatch boilerplate: discover backends, route
kwargs, merge docstrings, validate correctness. `scverse-backends` is
the shared machinery — host libraries depend on it, backends don't have
to.

## Contents

```{toctree}
:maxdepth: 2

usage/host
usage/backend
usage/conformance
release-notes
api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
