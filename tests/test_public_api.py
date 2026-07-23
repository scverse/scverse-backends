"""Tests for runtime introspection of the public typed API."""

from __future__ import annotations

from typing import get_type_hints

from scverse_backends import BackendDispatcher, Settings
from scverse_backends.testing import run_conformance


def test_public_type_hints_resolve_at_runtime():
    targets = [
        BackendDispatcher.__init__,
        BackendDispatcher.backend_dispatch.fget,
        BackendDispatcher.backend_class.fget,
        BackendDispatcher.settings.fget,
        BackendDispatcher.get_backend,
        BackendDispatcher.available_backend_names,
        BackendDispatcher.discover,
        Settings.__init__,
        Settings.backend.fget,
        Settings.backend.fset,
        Settings.use_backend,
        Settings.available_backends,
        Settings.get_backend,
        run_conformance,
    ]

    for target in targets:
        assert get_type_hints(target)
