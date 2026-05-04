"""Shared fixtures for scverse-backends tests."""

from __future__ import annotations

import pytest

from scverse_backends import BackendDispatcher


class FakeBackend:
    """Toy backend used across tests."""

    name = "fake_gpu"
    aliases = ["fake", "test-gpu"]

    def my_func(self, x, gpu_param=None):
        """Run my_func on the fake backend.

        Parameters
        ----------
        x
            Input value.
        gpu_param
            Backend-specific parameter.
        """
        return f"gpu:{x}:{gpu_param}"


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


def register_fake(d: BackendDispatcher) -> FakeBackend:
    """Register a ``FakeBackend`` directly on the registry, bypassing entrypoints."""
    backend = FakeBackend()
    d._registry._backends["fake_gpu"] = backend
    d._registry._alias_map["fake_gpu"] = "fake_gpu"
    d._registry._alias_map["fake"] = "fake_gpu"
    d._registry._alias_map["test-gpu"] = "fake_gpu"
    return backend
