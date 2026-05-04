"""Tests for the @dispatch decorator and lazy discovery."""

from __future__ import annotations

import inspect
import warnings

import pytest
from conftest import register_fake


class TestDispatch:
    def test_cpu_path(self, dispatcher):
        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}:{n_jobs}"

        assert my_func(42) == "cpu:42:None"
        assert my_func(42, n_jobs=4) == "cpu:42:4"

    def test_gpu_dispatch(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            assert my_func(42) == "gpu:42:None"

    def test_backend_specific_kwarg(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            assert my_func(42, gpu_param="hello") == "gpu:42:hello"

    def test_backend_none_uses_active_settings(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        dispatcher.settings.backend = "fake"
        assert my_func(42, backend=None) == "gpu:42:None"

    def test_backend_signature_default_is_none(self, dispatcher):
        @dispatcher.dispatch
        def my_func(x):
            return x

        sig = inspect.signature(my_func)
        assert sig.parameters["backend"].default is None

    def test_host_only_kwarg_warns(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=4)
                assert len(w) == 1
                assert "n_jobs" in str(w[0].message)

    def test_host_only_positional_arg_warns_and_is_not_forwarded(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}:{n_jobs}"

        with dispatcher.settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                assert my_func(42, 4) == "gpu:42:None"
                assert len(w) == 1
                assert "n_jobs" in str(w[0].message)

    def test_host_only_kwarg_default_silent(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=None)
                assert len(w) == 0

    def test_backend_kwarg_on_cpu_raises(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with pytest.raises(TypeError, match="gpu_param"):
            my_func(42, gpu_param="hello")

    def test_fallback_when_not_implemented(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def other_func(x):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            # FakeBackend doesn't have other_func -> CPU fallback
            assert other_func(42) == "cpu:42"

    def test_per_function_override(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        dispatcher.settings.backend = "fake"
        assert my_func(42, backend="cpu") == "cpu:42"

    def test_alias_resolution(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake_gpu"):
            assert my_func(42) == "gpu:42:None"
        with dispatcher.settings.use_backend("fake"):
            assert my_func(42) == "gpu:42:None"
        with dispatcher.settings.use_backend("test-gpu"):
            assert my_func(42) == "gpu:42:None"

    def test_backend_not_installed_raises(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ValueError, match="Unknown backend"):
            my_func(42, backend="nonexistent_backend")

    def test_per_call_trusted_backend_not_installed_has_install_hint(self, dispatcher):
        @dispatcher.dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ImportError, match="pip install fake-gpu-pkg"):
            my_func(42, backend="fake")

    def test_per_call_unknown_backend_suggests_close_name(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ValueError, match="Unknown backend.*Did you mean"):
            my_func(42, backend="fak")

    def test_reserved_backend_parameter_rejected(self, dispatcher):
        with pytest.raises(TypeError, match="reserved.*backend"):

            @dispatcher.dispatch
            def my_func(x, *, backend="old"):
                return x


class TestLazyDiscovery:
    def test_discovery_not_triggered_on_construction(self):
        """A fresh dispatcher should not have triggered discovery yet."""
        from scverse_backends import BackendDispatcher

        d = BackendDispatcher(entrypoint_group="nope.never", host_name="t")
        assert not d._registry._discovered

    def test_discovery_triggered_by_get_backend(self):
        from scverse_backends import BackendDispatcher

        d = BackendDispatcher(entrypoint_group="nope.never", host_name="t")
        d.get_backend("cpu")
        assert d._registry._discovered

    def test_discovery_triggered_by_settings_setter(self, dispatcher):
        register_fake(dispatcher)
        dispatcher._registry._discovered = False
        dispatcher.settings.backend = "fake"
        assert dispatcher._registry._discovered
