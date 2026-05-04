"""Shared fixtures for scverse-backends tests."""

from __future__ import annotations

import pytest

from scverse_backends import BackendDispatcher


@pytest.fixture
def dispatcher() -> BackendDispatcher:
    """Fresh dispatcher with ``fake_gpu`` listed as trusted but not yet registered."""
    d = BackendDispatcher(
        entrypoint_group="scverse_backends.test.no_such_group",
        host_name="testhost",
        trusted_backends={
            "fake_gpu": {
                "aliases": ["fake", "test-gpu"],
                "package": "fake-gpu-pkg",
            },
        },
    )
    # Mark as discovered so tests don't try to load real entrypoints.
    d._registry._discovered = True
    return d


@pytest.fixture
def untrusted_dispatcher() -> BackendDispatcher:
    """Dispatcher with no trusted backends — used to exercise the untrusted warning."""
    d = BackendDispatcher(
        entrypoint_group="scverse_backends.test.no_such_group",
        host_name="testhost",
        trusted_backends={},
    )
    d._registry._discovered = True
    return d
