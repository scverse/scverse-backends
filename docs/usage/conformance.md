# Feedback Testing Harness

`scverse-backends` includes a small optional runner for backend feedback
tests. It is not a required test framework and it does not define what
"correct" means for any scientific API.

The recommended pattern is:

- A host may expose reusable CPU-vs-backend checks for its own semantics.
- A backend may run those checks in its own CI for early compatibility
  feedback.
- `scverse-backends` only provides `run_conformance`, a tiny runner that
  resolves the backend, skips unimplemented functions, and reports results.

This keeps domain-specific correctness tests close to the host library,
while backend maintainers get a simple way to run the same checks before
opening or updating integrations.

## Host-Side Checks

Hosts can publish a small testing module with functions that compare the
host CPU reference against an active backend. The checks should be regular
pytest-style assertions and can use whatever fixtures or tolerances make
sense for that host.

```python
# example_host/testing/backend_feedback.py
from __future__ import annotations

import numpy as np


def _make_data(n_obs=500, n_vars=100):
    rng = np.random.default_rng(42)
    return rng.random((n_obs, n_vars), dtype=np.float32)


def _test_compute_score(backend_name: str) -> None:
    import example_host as eh

    data = _make_data()

    cpu = eh.compute_score(data.copy(), method="fast", copy=True)
    with eh.settings.use_backend(backend_name):
        backend = eh.compute_score(data.copy(), method="fast", copy=True)

    np.testing.assert_allclose(cpu, backend, atol=1e-5, rtol=1e-3)


CONFORMANCE_TESTS = {
    "compute_score": _test_compute_score,
}
```

The host decides whether this module is public API, testing API, or
experimental helper API. `scverse-backends` does not require a specific
location.

## Backend-Side Use

Backends can import the host's checks and run them in backend CI:

```python
# my_backend/tests/test_example_host_feedback.py
from __future__ import annotations

from scverse_backends.testing import run_conformance
from example_host._backends import get_backend
from example_host.testing.backend_feedback import CONFORMANCE_TESTS


def test_example_host_feedback():
    results = run_conformance(
        backend_name="my_backend",
        tests=CONFORMANCE_TESTS,
        get_backend=get_backend,
    )
    for function_name, status in results.items():
        assert status == "PASSED", f"{function_name}: {status}"
```

This is intended as fast feedback for backend maintainers. A host may
still keep its normal backend-independent unit tests, and a backend may
add deeper backend-specific tests around performance, memory behavior, or
hardware-specific edge cases.

## Runner Behavior

`run_conformance` does four small things:

1. Resolves `backend_name` using the host's `get_backend`.
2. Iterates over `{function_name: test_callable}` entries.
3. Skips functions the backend does not implement.
4. Records `"PASSED"`, `"SKIPPED (...)"`, or `"FAILED: ..."` and, by
   default, re-raises the first failure so pytest shows the useful
   traceback.

Passing `functions=[...]` restricts the run to named host checks. Unknown
function names and malformed filters raise `ValueError`, so a typo cannot
accidentally look like a successful empty run.

The runner intentionally does not import the host, NumPy, CuPy, or any
backend runtime. All scientific setup and assertions remain inside host-
or backend-owned tests.
