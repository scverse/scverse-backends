"""Test helpers shared across the suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def register_fake(d: BackendDispatcher) -> FakeBackend:
    """Register a ``FakeBackend`` directly on the registry, bypassing entrypoints."""
    backend = FakeBackend()
    d._registry._backends["fake_gpu"] = backend
    d._registry._alias_map["fake_gpu"] = "fake_gpu"
    d._registry._alias_map["fake"] = "fake_gpu"
    d._registry._alias_map["test-gpu"] = "fake_gpu"
    return backend
