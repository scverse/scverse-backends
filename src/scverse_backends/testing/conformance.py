"""Generic conformance test harness for scverse backends.

Hosts ship the actual test functions (CPU-vs-backend comparison logic);
this module provides only the runner.

Usage in a host's testing module::

    # squidpy/testing/backend_conformance.py
    from scverse_backends.testing import run_conformance
    from squidpy._backends import settings, get_backend

    def _test_spatial_autocorr(backend_name): ...
    def _test_co_occurrence(backend_name): ...

    _TESTS = {
        "spatial_autocorr": _test_spatial_autocorr,
        "co_occurrence": _test_co_occurrence,
    }

    def validate_backend(backend_name, functions=None):
        return run_conformance(
            backend_name=backend_name,
            tests=_TESTS,
            get_backend=get_backend,
            functions=functions,
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def run_conformance(
    *,
    backend_name: str,
    tests: dict[str, Callable[[str], None]],
    get_backend: Callable[[str], Any | None],
    functions: Sequence[str] | None = None,
    raise_on_failure: bool = True,
) -> dict[str, str]:
    """Run a host-supplied conformance suite against a backend.

    Parameters
    ----------
    backend_name
        Name or alias of the backend to test.
    tests
        Mapping ``{function_name: test_callable}``. The callable receives the
        canonical backend name and is expected to assert correctness internally
        (e.g. via ``np.testing.assert_allclose``).
    get_backend
        The host's ``get_backend`` function (typically
        ``dispatcher.get_backend``). Used to resolve aliases and to skip tests
        for functions the backend doesn't implement.
    functions
        Restrict the run to specific function names. ``None`` runs every test.
    raise_on_failure
        If True (default), re-raise the first test failure after recording it.

    Returns
    -------
    Dict mapping function name to ``"PASSED"``, ``"SKIPPED (...)"``, or
    ``"FAILED: ..."``.
    """
    backend = get_backend(backend_name)
    assert backend is not None, f"Backend {backend_name!r} not found"

    to_test = {k: v for k, v in tests.items() if functions is None or k in functions}

    results: dict[str, str] = {}
    for name, test_fn in to_test.items():
        method = getattr(backend, name, None)
        if method is None:
            results[name] = "SKIPPED (not implemented)"
            continue
        try:
            test_fn(backend_name)
            results[name] = "PASSED"
        except Exception as e:  # noqa: BLE001
            results[name] = f"FAILED: {e}"
            if raise_on_failure:
                raise
    return results
