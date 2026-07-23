"""Tests for the @backend_dispatch decorator and lazy discovery."""

from __future__ import annotations

import importlib.metadata
import inspect
import warnings
from dataclasses import dataclass

import pytest
from _helpers import register_fake


class TestDispatch:
    def test_cpu_path(self, dispatcher):
        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}:{n_jobs}"

        assert my_func(42) == "cpu:42:None"
        assert my_func(42, n_jobs=4) == "cpu:42:4"

    def test_gpu_dispatch(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            assert my_func(42) == "gpu:42:None"

    def test_backend_specific_kwarg(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            assert my_func(42, gpu_param="hello") == "gpu:42:hello"

    def test_shared_param_omitted_uses_backend_default(self, dispatcher):
        class BatchBackend:
            name = "batch_gpu"
            aliases = []

            def my_func(self, x, batch_size=4096):
                return f"gpu:{x}:{batch_size}"

        dispatcher._registry._backends["batch_gpu"] = BatchBackend()
        dispatcher._registry._alias_map["batch_gpu"] = "batch_gpu"
        dispatcher._registry._warned_untrusted.add("batch_gpu")

        @dispatcher.backend_dispatch
        def my_func(x, batch_size=None):
            return f"cpu:{x}:{batch_size}"

        with dispatcher.settings.use_backend("batch_gpu"):
            assert my_func(42) == "gpu:42:4096"
            assert my_func(42, batch_size=None) == "gpu:42:None"
            assert my_func(42, batch_size=8) == "gpu:42:8"

    def test_backend_only_param_named_args_is_allowed(self, dispatcher):
        class ArgsBackend:
            name = "args_gpu"
            aliases = []

            def my_func(self, x, args=None):
                return f"gpu:{x}:{args}"

        dispatcher._registry._backends["args_gpu"] = ArgsBackend()
        dispatcher._registry._alias_map["args_gpu"] = "args_gpu"
        dispatcher._registry._warned_untrusted.add("args_gpu")

        @dispatcher.backend_dispatch
        def my_func(x):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("args_gpu"):
            assert my_func(42, args="kept") == "gpu:42:kept"

    def test_backend_none_uses_active_settings(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        dispatcher.settings.backend = "fake"
        assert my_func(42, backend=None) == "gpu:42:None"

    def test_backend_signature_default_is_none(self, dispatcher):
        @dispatcher.backend_dispatch
        def my_func(x):
            return x

        sig = inspect.signature(my_func)
        assert sig.parameters["backend"].default is None

    def test_host_only_kwarg_warns(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=4)
                assert len(w) == 1
                assert "n_jobs" in str(w[0].message)

    def test_host_default_comparison_errors_do_not_break_dispatch(self, dispatcher):
        class ExplosiveDefault:
            def __eq__(self, other):
                raise RuntimeError("comparison is unavailable")

        class Backend:
            name = "comparison_gpu"
            aliases = []

            def my_func(self, x):
                return x

        dispatcher._registry._backends["comparison_gpu"] = Backend()
        dispatcher._registry._alias_map["comparison_gpu"] = "comparison_gpu"
        dispatcher._registry._warned_untrusted.add("comparison_gpu")
        default = ExplosiveDefault()

        @dispatcher.backend_dispatch
        def my_func(x, option=default):
            return x

        with dispatcher.settings.use_backend("comparison_gpu"):
            with pytest.warns(UserWarning, match="'option' has no effect"):
                assert my_func(1, option=ExplosiveDefault()) == 1

    def test_noncallable_backend_export_raises_clear_error(self, dispatcher):
        class Backend:
            name = "broken_gpu"
            aliases = []
            my_func = 42

        dispatcher._registry._backends["broken_gpu"] = Backend()
        dispatcher._registry._alias_map["broken_gpu"] = "broken_gpu"
        dispatcher._registry._warned_untrusted.add("broken_gpu")

        @dispatcher.backend_dispatch
        def my_func(x):
            return x

        with pytest.raises(TypeError, match="not callable"):
            my_func(1, backend="broken_gpu")

    def test_uninspectable_backend_export_raises_clear_error(self, dispatcher):
        class Backend:
            name = "opaque_gpu"
            aliases = []
            my_func = staticmethod(iter)

        dispatcher._registry._backends["opaque_gpu"] = Backend()
        dispatcher._registry._alias_map["opaque_gpu"] = "opaque_gpu"
        dispatcher._registry._warned_untrusted.add("opaque_gpu")

        @dispatcher.backend_dispatch
        def my_func(x):
            return x

        with pytest.raises(TypeError, match="must expose an inspectable signature"):
            my_func(object(), backend="opaque_gpu")

    def test_uninspectable_host_callable_is_rejected(self, dispatcher):
        with pytest.raises(TypeError, match="signature cannot be inspected"):
            dispatcher.backend_dispatch(iter)

    def test_runtime_functions_with_same_qualname_do_not_share_routing_cache(
        self,
        dispatcher,
    ):
        class Backend:
            name = "cache_gpu"
            aliases = []

            def f(self, x, backend_only=None):
                return f"gpu:{x}:{backend_only}"

        dispatcher._registry._backends["cache_gpu"] = Backend()
        dispatcher._registry._alias_map["cache_gpu"] = "cache_gpu"
        dispatcher._registry._warned_untrusted.add("cache_gpu")

        def make_first():
            @dispatcher.backend_dispatch
            def f(x, first=None):
                return f"cpu-first:{x}:{first}"

            return f

        def make_second():
            @dispatcher.backend_dispatch
            def f(x, second=None):
                return f"cpu-second:{x}:{second}"

            return f

        first = make_first()
        second = make_second()

        with dispatcher.settings.use_backend("cache_gpu"):
            with pytest.warns(UserWarning, match="first"):
                assert first(1, first="host") == "gpu:1:None"
            with pytest.warns(UserWarning, match="second"):
                assert second(1, second="host") == "gpu:1:None"

    def test_host_only_positional_arg_warns_and_is_not_forwarded(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
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

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=None)
                assert len(w) == 0

    def test_backend_kwarg_on_cpu_raises(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with pytest.raises(TypeError, match="gpu_param"):
            my_func(42, gpu_param="hello")

    def test_fallback_when_not_implemented(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def other_func(x):
            return f"cpu:{x}"

        with dispatcher.settings.use_backend("fake"):
            # FakeBackend doesn't have other_func -> CPU fallback
            assert other_func(42) == "cpu:42"

    def test_per_function_override(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        dispatcher.settings.backend = "fake"
        assert my_func(42, backend="cpu") == "cpu:42"

    def test_alias_resolution(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
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

        @dispatcher.backend_dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ValueError, match="Unknown backend"):
            my_func(42, backend="nonexistent_backend")

    def test_per_call_trusted_backend_not_installed_has_install_hint(self, dispatcher):
        @dispatcher.backend_dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ImportError, match="pip install fake-gpu-pkg"):
            my_func(42, backend="fake")

    def test_per_call_unknown_backend_suggests_close_name(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.backend_dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(ValueError, match="Unknown backend.*Did you mean"):
            my_func(42, backend="fak")

    def test_reserved_backend_parameter_rejected(self, dispatcher):
        with pytest.raises(TypeError, match="reserved.*backend"):

            @dispatcher.backend_dispatch
            def my_func(x, *, backend="old"):
                return x

    def test_classes_and_methods_are_rejected(self, dispatcher):
        class Model:
            pass

        with pytest.raises(TypeError, match="classes are not supported"):
            dispatcher.backend_dispatch(Model)

        def method(self, value):
            return value

        with pytest.raises(TypeError, match="module-level functions"):
            dispatcher.backend_dispatch(method)

        def class_method(cls, value):
            return value

        with pytest.raises(TypeError, match="module-level functions"):
            dispatcher.backend_dispatch(class_method)

        class WithMethod:
            def method(self, value):
                return value

        with pytest.raises(TypeError, match="module-level functions"):
            dispatcher.backend_dispatch(WithMethod().method)

    def test_variadic_positional_host_parameter_is_rejected(self, dispatcher):
        def my_func(x, *values):
            return x, values

        with pytest.raises(TypeError, match=r"variadic positional.*\\*args"):
            dispatcher.backend_dispatch(my_func)

    def test_ambiguous_default_comparison_warns_instead_of_crashing(self, dispatcher):
        class AmbiguousTruth:
            def __bool__(self):
                raise ValueError("ambiguous")

        class DefaultValue:
            def __eq__(self, other):
                return AmbiguousTruth()

        class Backend:
            name = "default_gpu"
            aliases = []

            def my_func(self, x):
                return x

        backend = Backend()
        dispatcher._registry._backends[backend.name] = backend
        dispatcher._registry._alias_map[backend.name] = backend.name
        dispatcher._registry._warned_untrusted.add(backend.name)
        default = DefaultValue()

        @dispatcher.backend_dispatch
        def my_func(x, option=default):
            return x, option

        with dispatcher.settings.use_backend(backend.name):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert my_func(1, option=default) == 1
                assert caught == []

            with pytest.warns(UserWarning, match="option"):
                assert my_func(2, option=DefaultValue()) == 2

    def test_positional_only_backend_defaults_are_not_shifted(self, dispatcher):
        class Backend:
            name = "positional_gpu"
            aliases = []

            def my_func(self, prefix="backend-default", x=None, /):
                return prefix, x

        backend = Backend()
        dispatcher._registry._backends[backend.name] = backend
        dispatcher._registry._alias_map[backend.name] = backend.name
        dispatcher._registry._warned_untrusted.add(backend.name)

        @dispatcher.backend_dispatch
        def my_func(x):
            return "cpu", x

        assert my_func(4, backend=backend.name) == ("backend-default", 4)

    def test_required_positional_only_backend_prefix_has_clear_error(self, dispatcher):
        class Backend:
            name = "positional_gpu"
            aliases = []

            def my_func(self, prefix, x, /):
                return prefix, x

        backend = Backend()
        dispatcher._registry._backends[backend.name] = backend
        dispatcher._registry._alias_map[backend.name] = backend.name
        dispatcher._registry._warned_untrusted.add(backend.name)

        @dispatcher.backend_dispatch
        def my_func(x):
            return "cpu", x

        with pytest.raises(TypeError, match="required positional-only.*prefix"):
            my_func(4, backend=backend.name)

    def test_positional_only_and_var_keyword_routing(self, dispatcher):
        class Backend:
            name = "flexible_gpu"
            aliases = []

            def my_func(self, x, /, gpu_param=None, **kwargs):
                return x, gpu_param, kwargs

        backend = Backend()
        dispatcher._registry._backends[backend.name] = backend
        dispatcher._registry._alias_map[backend.name] = backend.name
        dispatcher._registry._warned_untrusted.add(backend.name)

        @dispatcher.backend_dispatch
        def my_func(x, /, **kwargs):
            return x, kwargs

        assert my_func(
            4,
            gpu_param="specific",
            extra="forwarded",
            backend=backend.name,
        ) == (4, "specific", {"extra": "forwarded"})


class TestBackendClass:
    def test_cpu_class_is_constructed_by_default(self, dispatcher):
        @dispatcher.backend_class
        class Model:
            """A CPU model."""

            def __init__(self, value):
                self.value = value

        model = Model("cpu")

        assert model.value == "cpu"
        assert isinstance(model, Model)
        assert str(inspect.signature(Model)) == "(value, *, backend: str | None = None)"
        assert Model.__scverse_backends_cpu_class__.__name__ == "Model"
        assert Model.__doc__ == "A CPU model."

    def test_active_backend_replaces_entire_class(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

            class Model:
                def __init__(self, value, *, device="gpu"):
                    self.value = value
                    self.device = device

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            def __init__(self, value, *, device="cpu"):
                self.value = value
                self.device = device

        with dispatcher.settings.use_backend("class_gpu"):
            model = Model("accelerated", device="cuda")

        assert isinstance(model, ClassBackend.Model)
        assert model.value == "accelerated"
        assert model.device == "cuda"

    def test_backend_none_uses_active_setting(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

            class Model:
                source = "gpu"

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            source = "cpu"

        with dispatcher.settings.use_backend("class_gpu"):
            assert Model(backend=None).source == "gpu"

    def test_per_instance_backend_override(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

            class Model:
                def __init__(self, value):
                    self.value = f"gpu:{value}"

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            def __init__(self, value):
                self.value = f"cpu:{value}"

        assert Model("one").value == "cpu:one"
        assert Model("two", backend="class_gpu").value == "gpu:two"
        assert Model("three", backend="cpu").value == "cpu:three"

    def test_backend_without_matching_class_falls_back_to_cpu(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            def __init__(self, value):
                self.value = value

        with dispatcher.settings.use_backend("class_gpu"):
            model = Model("cpu fallback")

        assert model.value == "cpu fallback"
        assert isinstance(model, Model)

    def test_undecorated_subclass_does_not_dispatch_as_parent(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

            class Model:
                source = "gpu"

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            source = "cpu"

        class SpecializedModel(Model):
            source = "specialized"

            def __init__(self, *, backend):
                self.constructor_backend = backend

        with dispatcher.settings.use_backend("class_gpu"):
            model = SpecializedModel(backend="application-value")

        assert isinstance(model, SpecializedModel)
        assert model.source == "specialized"
        assert model.constructor_backend == "application-value"

    def test_decorated_subclass_dispatches_by_its_own_name(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

            class Parent:
                source = "gpu-parent"

            class Child:
                source = "gpu-child"

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Parent:
            source = "cpu-parent"

        @dispatcher.backend_class
        class Child(Parent):
            source = "cpu-child"

        with dispatcher.settings.use_backend("class_gpu"):
            assert Parent().source == "gpu-parent"
            assert Child().source == "gpu-child"

    def test_custom_metaclass_is_preserved(self, dispatcher):
        calls: list[str] = []

        class TrackingMeta(type):
            def __call__(cls, *args, **kwargs):
                calls.append(cls.__name__)
                return super().__call__(*args, **kwargs)

        @dispatcher.backend_class
        class Model(metaclass=TrackingMeta):
            pass

        model = Model()

        assert isinstance(type(Model), type)
        assert isinstance(Model, TrackingMeta)
        assert isinstance(model, Model)
        assert calls == ["Model"]

    def test_custom_metaclass_prepared_namespace_is_preserved(self, dispatcher):
        class PreparedNamespace(dict):
            pass

        class PreparedMeta(type):
            @classmethod
            def __prepare__(mcls, name, bases):
                return PreparedNamespace()

            def __new__(mcls, name, bases, namespace):
                if not isinstance(namespace, PreparedNamespace):
                    raise TypeError("prepared namespace required")
                return super().__new__(mcls, name, bases, namespace)

        @dispatcher.backend_class
        class Model(metaclass=PreparedMeta):
            pass

        assert isinstance(Model(), Model)

    def test_class_api_and_slots_are_preserved(self, dispatcher):
        @dispatcher.backend_class
        class Model:
            """A model."""

            __slots__ = ("value",)
            category = "host"

            def __init__(self, value):
                self.value = value

            @classmethod
            def class_name(cls):
                return cls.__name__

        model = Model(3)

        assert model.value == 3
        assert Model.category == "host"
        assert Model.class_name() == "Model"
        assert Model.__doc__ == "A model."
        with pytest.raises(AttributeError):
            model.extra = 1

    def test_dataclass_behavior_is_preserved(self, dispatcher):
        @dispatcher.backend_class
        @dataclass(frozen=True)
        class Model:
            value: int

        assert Model(3) == Model(3)
        assert Model(3).value == 3

    def test_backend_implementation_may_inherit_from_host_class(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

        backend = ClassBackend()
        dispatcher._registry._backends["class_gpu"] = backend
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            source = "cpu"

        class BackendModel(Model):
            source = "gpu"

        backend.Model = BackendModel

        model = Model(backend="class_gpu")

        assert isinstance(model, Model)
        assert isinstance(model, BackendModel)
        assert model.source == "gpu"

    def test_backend_attribute_must_be_a_class(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []
            Model = object()

        dispatcher._registry._backends["class_gpu"] = ClassBackend()
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            pass

        with pytest.raises(TypeError, match="is not a class"):
            Model(backend="class_gpu")

    def test_backend_cannot_point_back_to_dispatched_host_class(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

        backend = ClassBackend()
        dispatcher._registry._backends["class_gpu"] = backend
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            pass

        backend.Model = Model

        with pytest.raises(TypeError, match="dispatched host class"):
            Model(backend="class_gpu")

    def test_original_host_class_exported_by_backend_falls_back(self, dispatcher):
        class ClassBackend:
            name = "class_gpu"
            aliases = []

        backend = ClassBackend()
        dispatcher._registry._backends["class_gpu"] = backend
        dispatcher._registry._alias_map["class_gpu"] = "class_gpu"
        dispatcher._registry._warned_untrusted.add("class_gpu")

        @dispatcher.backend_class
        class Model:
            source = "cpu"

        backend.Model = Model.__scverse_backends_cpu_class__

        model = Model(backend="class_gpu")

        assert isinstance(model, Model)
        assert model.source == "cpu"

    def test_double_decoration_is_rejected(self, dispatcher):
        @dispatcher.backend_class
        class Model:
            pass

        with pytest.raises(TypeError, match="already decorated"):
            dispatcher.backend_class(Model)

    def test_non_subclassable_host_class_has_clear_error(self, dispatcher):
        class RequiresClassOption:
            def __init_subclass__(cls, *, enabled, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.enabled = enabled

        class Model(RequiresClassOption, enabled=True):
            pass

        with pytest.raises(TypeError, match="requires the host class to support"):
            dispatcher.backend_class(Model)

    def test_non_class_and_reserved_constructor_parameter_are_rejected(
        self, dispatcher
    ):
        with pytest.raises(TypeError, match="only decorate classes"):
            dispatcher.backend_class(lambda: None)

        with pytest.raises(TypeError, match="reserved.*backend"):

            @dispatcher.backend_class
            class Model:
                def __init__(self, backend):
                    self.backend = backend


class TestLazyDiscovery:
    def test_discovery_not_triggered_on_construction(self):
        """A fresh dispatcher should not have triggered discovery yet."""
        from scverse_backends import BackendDispatcher

        d = BackendDispatcher(entrypoint_group="nope.never", host_name="t")
        assert not d._registry._discovered

    def test_cpu_selection_does_not_trigger_discovery(self):
        from scverse_backends import BackendDispatcher

        d = BackendDispatcher(entrypoint_group="nope.never", host_name="t")

        assert d.get_backend("cpu") is None
        d.settings.backend = "cpu"
        with d.settings.use_backend("cpu"):
            assert d.settings.backend == "cpu"

        assert not d._registry._discovered

    def test_discovery_triggered_by_non_cpu_get_backend(self):
        from scverse_backends import BackendDispatcher

        d = BackendDispatcher(entrypoint_group="nope.never", host_name="t")
        d.get_backend("missing")

        assert d._registry._discovered

    def test_discovery_triggered_by_settings_setter(self, dispatcher):
        register_fake(dispatcher)
        dispatcher._registry._discovered = False
        dispatcher.settings.backend = "fake"
        assert dispatcher._registry._discovered

    def test_discovery_without_backends_updates_generated_docs(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda *, group: [],
        )
        dispatcher = BackendDispatcher(
            entrypoint_group="empty.backends",
            host_name="empty",
        )

        @dispatcher.backend_dispatch
        def my_func(x):
            """Run the function.

            Parameters
            ----------
            x
                Input value.
            """
            return x

        dispatcher.discover()

        assert "Backend selector injected" in my_func.__doc__

    def test_backend_class_override_triggers_discovery(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        class ClassBackend:
            name = "class_gpu"
            aliases = ["class"]

            class Model:
                source = "gpu"

        class Entrypoint:
            name = "class_gpu"
            value = f"{__name__}:ClassBackend"
            dist = None

            def load(self):
                return ClassBackend()

        def entry_points(*, group):
            assert group == "class.backends"
            return [Entrypoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
        dispatcher = BackendDispatcher(
            entrypoint_group="class.backends",
            host_name="test",
        )

        @dispatcher.backend_class
        class Model:
            source = "cpu"

        assert not dispatcher._registry._discovered
        with pytest.warns(UserWarning, match="not in test's trusted"):
            model = Model(backend="class")

        assert dispatcher._registry._discovered
        assert isinstance(model, ClassBackend.Model)
