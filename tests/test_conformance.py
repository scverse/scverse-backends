"""Tests for the feedback testing harness."""

from __future__ import annotations

import types

import pytest

from scverse_backends.testing import run_conformance


def test_run_conformance_passes_and_skips_missing_functions():
    backend = types.SimpleNamespace(implemented=object())
    calls: list[str] = []

    def get_backend(name: str):
        assert name == "cuda"
        return backend

    def implemented_test(name: str) -> None:
        calls.append(name)

    def missing_test(name: str) -> None:
        calls.append(name)

    results = run_conformance(
        backend_name="cuda",
        tests={
            "implemented": implemented_test,
            "missing": missing_test,
        },
        get_backend=get_backend,
    )

    assert results == {
        "implemented": "PASSED",
        "missing": "SKIPPED (not implemented)",
    }
    assert calls == ["cuda"]


def test_run_conformance_can_limit_functions():
    backend = types.SimpleNamespace(implemented=object())

    def get_backend(name: str):
        return backend

    results = run_conformance(
        backend_name="cuda",
        tests={
            "implemented": lambda name: None,
            "other": lambda name: None,
        },
        get_backend=get_backend,
        functions=["implemented"],
    )

    assert results == {"implemented": "PASSED"}


def test_run_conformance_records_failures_without_raising():
    backend = types.SimpleNamespace(implemented=object())

    def get_backend(name: str):
        return backend

    def failing_test(name: str) -> None:
        raise RuntimeError("boom")

    results = run_conformance(
        backend_name="cuda",
        tests={"implemented": failing_test},
        get_backend=get_backend,
        raise_on_failure=False,
    )

    assert results == {"implemented": "FAILED: boom"}


def test_run_conformance_raises_failures_by_default():
    backend = types.SimpleNamespace(implemented=object())

    def get_backend(name: str):
        return backend

    def failing_test(name: str) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_conformance(
            backend_name="cuda",
            tests={"implemented": failing_test},
            get_backend=get_backend,
        )


def test_run_conformance_requires_backend():
    with pytest.raises(AssertionError, match="Backend 'cuda' not found"):
        run_conformance(
            backend_name="cuda",
            tests={"implemented": lambda name: None},
            get_backend=lambda name: None,
        )
