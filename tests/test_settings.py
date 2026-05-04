"""Tests for the per-host settings/registry."""

from __future__ import annotations

import warnings

import pytest

from tests.conftest import register_fake


class TestSettings:
    def test_default_is_cpu(self, dispatcher):
        assert dispatcher.settings.backend == "cpu"

    def test_set_unknown_backend_raises_with_suggestion(self, dispatcher):
        register_fake(dispatcher)
        with pytest.raises(ValueError, match="Unknown backend.*Did you mean"):
            dispatcher.settings.backend = "fak"  # close to "fake"

    def test_set_unknown_backend_raises(self, dispatcher):
        with pytest.raises(ValueError, match="Unknown backend"):
            dispatcher.settings.backend = "nonexistent"

    def test_trusted_but_not_installed_raises(self, dispatcher):
        # fake_gpu is trusted but not registered (not installed)
        with pytest.raises(ImportError, match="not installed"):
            dispatcher.settings.backend = "fake"

    def test_context_manager_restores(self, dispatcher):
        register_fake(dispatcher)

        dispatcher.settings.backend = "fake"
        with dispatcher.settings.use_backend("cpu"):
            assert dispatcher.settings.backend == "cpu"
        assert dispatcher.settings.backend == "fake_gpu"

    def test_set_via_alias(self, dispatcher):
        register_fake(dispatcher)
        dispatcher.settings.backend = "fake"
        assert dispatcher.settings.backend == "fake_gpu"

    def test_set_via_canonical(self, dispatcher):
        register_fake(dispatcher)
        dispatcher.settings.backend = "fake_gpu"
        assert dispatcher.settings.backend == "fake_gpu"

    def test_untrusted_backend_warns(self, untrusted_dispatcher):
        register_fake(untrusted_dispatcher)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untrusted_dispatcher.settings.backend = "fake_gpu"
            assert len(w) == 1
            assert "not in testhost's trusted backends list" in str(w[0].message)

    def test_available_backends_empty(self, dispatcher):
        assert dispatcher.settings.available_backends() == []

    def test_available_backends_with_registered(self, dispatcher):
        register_fake(dispatcher)
        assert "fake_gpu" in dispatcher.settings.available_backends()

    def test_get_backend_returns_instance(self, dispatcher):
        backend = register_fake(dispatcher)
        assert dispatcher.settings.get_backend("fake_gpu") is backend

    def test_get_backend_by_alias(self, dispatcher):
        backend = register_fake(dispatcher)
        assert dispatcher.settings.get_backend("fake") is backend
        assert dispatcher.settings.get_backend("test-gpu") is backend

    def test_get_backend_unknown_returns_none(self, dispatcher):
        assert dispatcher.settings.get_backend("nonexistent") is None

    def test_get_backend_cpu_returns_none(self, dispatcher):
        assert dispatcher.settings.get_backend("cpu") is None

    def test_two_dispatchers_isolated(self):
        """Two dispatchers in the same process must not share settings state."""
        from scverse_backends import BackendDispatcher

        a = BackendDispatcher(entrypoint_group="a.x", host_name="a")
        b = BackendDispatcher(entrypoint_group="b.x", host_name="b")
        a._registry._discovered = True
        b._registry._discovered = True

        register_fake(a)
        a.settings.backend = "fake_gpu"
        # b's default is still cpu and unaffected
        assert b.settings.backend == "cpu"
