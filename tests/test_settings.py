"""Tests for the per-host settings/registry."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.metadata
import threading
import time
import types
import warnings

import pytest
from _helpers import register_fake


class TestSettings:
    def test_default_is_cpu(self, dispatcher):
        assert dispatcher.settings.backend == "cpu"

    def test_settings_type_is_public(self, dispatcher):
        from scverse_backends import Settings

        assert isinstance(dispatcher.settings, Settings)

    def test_set_unknown_backend_raises_with_suggestion(self, dispatcher):
        register_fake(dispatcher)
        with pytest.raises(ValueError, match="Unknown backend.*Did you mean"):
            dispatcher.settings.backend = "fak"  # close to "fake"

    def test_set_unknown_backend_raises(self, dispatcher):
        with pytest.raises(ValueError, match="Unknown backend"):
            dispatcher.settings.backend = "nonexistent"

    def test_set_non_string_backend_raises_clear_value_error(self, dispatcher):
        with pytest.raises(ValueError, match="Backend name must be"):
            dispatcher.settings.backend = None

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

    def test_untrusted_backend_warns_once(self, untrusted_dispatcher):
        register_fake(untrusted_dispatcher)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untrusted_dispatcher.settings.backend = "fake_gpu"
            untrusted_dispatcher.settings.backend = "cpu"
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

        trusted = {"fake_gpu": {"aliases": ["fake"], "package": "fake-pkg"}}
        a = BackendDispatcher(
            entrypoint_group="a.x", host_name="a", trusted_backends=trusted
        )
        b = BackendDispatcher(
            entrypoint_group="b.x", host_name="b", trusted_backends=trusted
        )
        a._registry._discovered = True
        b._registry._discovered = True

        register_fake(a)
        a.settings.backend = "fake_gpu"
        # b's default is still cpu and unaffected
        assert b.settings.backend == "cpu"
        assert a.settings.backend == "fake_gpu"

    def test_threads_can_hold_different_active_backends(self, dispatcher):
        register_fake(dispatcher)
        ready = threading.Barrier(2)
        release = threading.Barrier(2)

        def worker(name: str) -> str:
            with dispatcher.settings.use_backend(name):
                ready.wait(timeout=5)
                value = dispatcher.settings.backend
                release.wait(timeout=5)
                return value

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fake_future = executor.submit(worker, "fake")
            cpu_future = executor.submit(worker, "cpu")

        assert fake_future.result() == "fake_gpu"
        assert cpu_future.result() == "cpu"
        assert dispatcher.settings.backend == "cpu"

    def test_async_tasks_can_hold_different_active_backends(self, dispatcher):
        register_fake(dispatcher)

        async def worker(name: str) -> str:
            with dispatcher.settings.use_backend(name):
                await asyncio.sleep(0)
                return dispatcher.settings.backend

        async def main() -> tuple[str, str]:
            fake_backend, cpu_backend = await asyncio.gather(
                worker("fake"),
                worker("cpu"),
            )
            return fake_backend, cpu_backend

        assert asyncio.run(main()) == ("fake_gpu", "cpu")
        assert dispatcher.settings.backend == "cpu"

    def test_discovery_is_locked_across_threads(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        load_count = 0
        load_count_lock = threading.Lock()

        class Backend:
            name = "thread_gpu"
            aliases = []

        class Entrypoint:
            name = "thread_gpu"
            value = "thread_pkg:Backend"
            dist = None

            def load(self):
                nonlocal load_count
                with load_count_lock:
                    load_count += 1
                time.sleep(0.05)
                return Backend()

        def entry_points(*, group):
            assert group == "thread.backends"
            return [Entrypoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
        dispatcher = BackendDispatcher(
            entrypoint_group="thread.backends",
            host_name="threadhost",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(lambda _: dispatcher.available_backend_names(), range(2))
            )

        assert results == [["thread_gpu"], ["thread_gpu"]]
        assert load_count == 1

    def test_trusted_alias_cannot_be_claimed_by_untrusted_backend(self, dispatcher):
        class BadBackend:
            name = "bad_gpu"
            aliases = ["fake"]

        with pytest.warns(UserWarning, match="reserved for trusted backend"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert dispatcher.settings.get_backend("bad_gpu").name == "bad_gpu"
        with pytest.raises(ImportError, match="fake-gpu-pkg"):
            dispatcher.settings.backend = "fake"

    def test_invalid_backend_name_is_ignored(self, dispatcher):
        class BadBackend:
            name = "bad\nname"
            aliases = []

        with pytest.warns(UserWarning, match="safe backend label"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert "bad\nname" not in dispatcher.available_backend_names()

    def test_invalid_backend_alias_is_ignored(self, dispatcher):
        class BadBackend:
            name = "bad_gpu"
            aliases = ["bad alias"]

        with pytest.warns(UserWarning, match="safe backend labels"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert "bad_gpu" in dispatcher.available_backend_names()
        assert "bad alias" not in dispatcher.available_backend_names()

    def test_string_backend_aliases_are_rejected_as_invalid_config(self, dispatcher):
        class BadBackend:
            name = "my_gpu"
            aliases = "cuda"

        with pytest.warns(UserWarning, match="backend.aliases must be"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert dispatcher.available_backend_names() == ["my_gpu"]
        for char in "cuda":
            assert char not in dispatcher.available_backend_names()

    def test_none_backend_aliases_are_rejected_as_invalid_config(self, dispatcher):
        class BadBackend:
            name = "my_gpu"
            aliases = None

        with pytest.warns(UserWarning, match="not None"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert dispatcher.available_backend_names() == ["my_gpu"]

    def test_invalid_trusted_alias_raises(self):
        from scverse_backends import BackendDispatcher

        with pytest.raises(ValueError, match="Invalid trusted backend alias"):
            BackendDispatcher(
                entrypoint_group="nope.never",
                host_name="testhost",
                trusted_backends={
                    "fake_gpu": {"aliases": ["bad alias"], "package": "fake-pkg"}
                },
            )

    def test_trusted_alias_string_raises(self):
        from scverse_backends import BackendDispatcher

        with pytest.raises(ValueError, match="aliases.*list, tuple, or set"):
            BackendDispatcher(
                entrypoint_group="nope.never",
                host_name="testhost",
                trusted_backends={
                    "fake_gpu": {"aliases": "fake", "package": "fake-pkg"}
                },
            )

    def test_duplicate_trusted_alias_raises(self):
        from scverse_backends import BackendDispatcher

        with pytest.raises(ValueError, match="configured for both"):
            BackendDispatcher(
                entrypoint_group="nope.never",
                host_name="testhost",
                trusted_backends={
                    "first": {"aliases": ["shared"], "package": "first-pkg"},
                    "second": {"aliases": ["shared"], "package": "second-pkg"},
                },
            )

    def test_reserved_backend_name_raises_on_use(self):
        from scverse_backends import BackendDispatcher

        dispatcher = BackendDispatcher(
            entrypoint_group="nope.never",
            host_name="testhost",
            reserved_backends={"gpu": "Use a concrete backend alias."},
        )
        dispatcher._registry._discovered = True

        with pytest.raises(ValueError, match="Use a concrete backend alias"):
            dispatcher.settings.backend = "gpu"

    def test_reserved_backend_name_cannot_be_registered(self):
        from scverse_backends import BackendDispatcher

        class BadBackend:
            name = "gpu"
            aliases = []

        dispatcher = BackendDispatcher(
            entrypoint_group="nope.never",
            host_name="testhost",
            reserved_backends={"gpu": "Use a concrete backend alias."},
        )
        dispatcher._registry._discovered = True

        with pytest.warns(UserWarning, match="reserved by testhost"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert dispatcher.settings.get_backend("gpu") is None

    def test_reserved_backend_alias_cannot_be_registered(self):
        from scverse_backends import BackendDispatcher

        class BadBackend:
            name = "bad_gpu"
            aliases = ["gpu"]

        dispatcher = BackendDispatcher(
            entrypoint_group="nope.never",
            host_name="testhost",
            reserved_backends={"gpu": "Use a concrete backend alias."},
        )
        dispatcher._registry._discovered = True

        with pytest.warns(UserWarning, match="reserved by testhost"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert dispatcher.settings.get_backend("bad_gpu").name == "bad_gpu"
        with pytest.raises(ValueError, match="Use a concrete backend alias"):
            dispatcher.settings.backend = "gpu"

    def test_reserved_backend_conflicts_with_trusted_alias(self):
        from scverse_backends import BackendDispatcher

        with pytest.raises(ValueError, match="reserved by testhost"):
            BackendDispatcher(
                entrypoint_group="nope.never",
                host_name="testhost",
                trusted_backends={
                    "fake_gpu": {"aliases": ["gpu"], "package": "fake-pkg"}
                },
                reserved_backends={"gpu": "Use a concrete backend alias."},
            )

    def test_trusted_canonical_requires_expected_distribution(self, dispatcher):
        class BadBackend:
            name = "fake_gpu"
            aliases = ["fake"]

        with pytest.warns(UserWarning, match="expected distribution fake-gpu-pkg"):
            dispatcher._registry._register_backend(
                BadBackend(),
                entrypoint_name="fake_gpu",
                distribution_name="evil-pkg",
            )

        assert dispatcher.settings.get_backend("fake_gpu") is None
        with pytest.raises(ImportError, match="Refusing trusted backend"):
            dispatcher.settings.backend = "fake"

    def test_trusted_canonical_accepts_expected_distribution(self, dispatcher):
        class GoodBackend:
            name = "fake_gpu"
            aliases = ["fake"]

        dispatcher._registry._register_backend(
            GoodBackend(),
            entrypoint_name="fake_gpu",
            distribution_name="fake-gpu-pkg",
        )

        assert dispatcher.settings.get_backend("fake").name == "fake_gpu"
        dispatcher.settings.backend = "fake"
        assert dispatcher.settings.backend == "fake_gpu"

    def test_trusted_canonical_can_require_entrypoint_name(self):
        from scverse_backends import BackendDispatcher

        class GoodBackend:
            name = "fake_gpu"
            aliases = []

        dispatcher = BackendDispatcher(
            entrypoint_group="nope.never",
            host_name="testhost",
            trusted_backends={
                "fake_gpu": {
                    "aliases": [],
                    "package": "fake-gpu-pkg",
                    "entrypoints": ["expected"],
                },
            },
        )
        dispatcher._registry._discovered = True

        with pytest.warns(UserWarning, match="expected one of"):
            dispatcher._registry._register_backend(
                GoodBackend(),
                entrypoint_name="wrong",
                distribution_name="fake-gpu-pkg",
            )

        with pytest.raises(ImportError, match="Refusing trusted backend"):
            dispatcher.settings.backend = "fake_gpu"

    def test_trusted_canonical_can_require_module_prefix(self):
        from scverse_backends import BackendDispatcher

        class GoodBackend:
            name = "fake_gpu"
            aliases = []

        GoodBackend.__module__ = "allowed_pkg.adapter"

        dispatcher = BackendDispatcher(
            entrypoint_group="nope.never",
            host_name="testhost",
            trusted_backends={
                "fake_gpu": {
                    "aliases": [],
                    "package": "fake-gpu-pkg",
                    "module_prefixes": ["allowed_pkg"],
                },
            },
        )
        dispatcher._registry._discovered = True

        dispatcher._registry._register_backend(
            GoodBackend(),
            entrypoint_name="fake_gpu",
            distribution_name="fake-gpu-pkg",
        )

        assert dispatcher.settings.get_backend("fake_gpu").name == "fake_gpu"

    def test_class_entrypoint_is_instantiated(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        class GoodBackend:
            name = "fake_gpu"
            aliases = ["fake"]

        class FakeDistribution:
            metadata = {"Name": "fake-gpu-pkg"}

        class GoodEntrypoint:
            name = "fake_gpu"
            value = "allowed_pkg.adapter:GoodBackend"
            dist = FakeDistribution()

            def load(self):
                return GoodBackend

        def entry_points(*, group):
            assert group == "class.backends"
            return [GoodEntrypoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
        dispatcher = BackendDispatcher(
            entrypoint_group="class.backends",
            host_name="testhost",
            trusted_backends={
                "fake_gpu": {
                    "aliases": ["fake"],
                    "package": "fake-gpu-pkg",
                    "module_prefixes": [__name__],
                },
            },
        )

        backend = dispatcher.settings.get_backend("fake")

        assert isinstance(backend, GoodBackend)

    def test_module_entrypoint_is_registered_directly(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        module = types.ModuleType("allowed_pkg.adapter")
        module.name = "fake_gpu"
        module.aliases = ["fake"]

        def my_func(x, gpu_param=None):
            return f"module:{x}:{gpu_param}"

        module.my_func = my_func

        class FakeDistribution:
            metadata = {"Name": "fake-gpu-pkg"}

        class GoodEntrypoint:
            name = "fake_gpu"
            value = "allowed_pkg.adapter"
            dist = FakeDistribution()

            def load(self):
                return module

        def entry_points(*, group):
            assert group == "module.backends"
            return [GoodEntrypoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
        dispatcher = BackendDispatcher(
            entrypoint_group="module.backends",
            host_name="testhost",
            trusted_backends={
                "fake_gpu": {
                    "aliases": ["fake"],
                    "package": "fake-gpu-pkg",
                    "module_prefixes": ["allowed_pkg"],
                },
            },
        )

        assert dispatcher.settings.get_backend("fake") is module
        dispatcher.settings.backend = "fake"
        assert dispatcher.settings.backend == "fake_gpu"

    def test_cpu_alias_is_reserved(self, dispatcher):
        class BadBackend:
            name = "bad_gpu"
            aliases = ["cpu"]

        with pytest.warns(UserWarning, match="reserved"):
            dispatcher._registry._register_backend(
                BadBackend(), entrypoint_name="bad_backend"
            )

        assert "bad_gpu" in dispatcher.available_backend_names()
        assert "cpu" not in dispatcher.available_backend_names()
        assert dispatcher.get_backend("cpu") is None

    def test_failed_entrypoint_error_is_reported(self, monkeypatch):
        from scverse_backends import BackendDispatcher

        class BrokenEntrypoint:
            name = "broken_gpu"

            def load(self):
                raise RuntimeError("boom")

        def entry_points(*, group):
            assert group == "broken.backends"
            return [BrokenEntrypoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
        dispatcher = BackendDispatcher(
            entrypoint_group="broken.backends",
            host_name="broken",
        )

        with pytest.raises(ImportError, match="failed to load") as exc_info:
            dispatcher.settings.backend = "broken_gpu"

        assert isinstance(exc_info.value.__cause__, RuntimeError)
