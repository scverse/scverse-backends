"""End-to-end tests for signature & docstring merging.

The unit tests in ``test_docstring.py`` cover the numpydoc helpers in
isolation. These tests cover the full pipeline: a function decorated with
``@dispatch``, a backend registered against the dispatcher, ``discover()``
called, and the wrapper's ``__signature__`` and ``__doc__`` checked.
"""

from __future__ import annotations

import inspect

import pytest
from _helpers import register_fake


def _trigger_merge(dispatcher):
    """Force the merge to run without going through entrypoint loading."""
    dispatcher._dispatch_impl._update_signatures()


class TestSignatureMerge:
    def test_backend_kwarg_injected_at_decoration(self, dispatcher):
        @dispatcher.dispatch
        def f(x, *, mode="moran"):
            return x

        sig = inspect.signature(f)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default is None

    def test_backend_only_param_merged_into_signature(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return x

        _trigger_merge(dispatcher)

        sig = inspect.signature(my_func)
        # Original host params preserved
        assert "x" in sig.parameters
        assert "n_jobs" in sig.parameters
        # FakeBackend.my_func(self, x, gpu_param=None) — gpu_param is backend-only
        assert "gpu_param" in sig.parameters
        # backend is still present and last among non-VAR params
        assert "backend" in sig.parameters

    def test_merge_is_idempotent(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            """Run my_func.

            Parameters
            ----------
            x
                Input.
            n_jobs
                Host-only.
            """
            return x

        _trigger_merge(dispatcher)
        first_doc = my_func.__doc__
        first_sig = str(inspect.signature(my_func))

        # Run the merge again — must not double-inject
        _trigger_merge(dispatcher)
        assert my_func.__doc__ == first_doc
        assert str(inspect.signature(my_func)) == first_sig

        # And again
        _trigger_merge(dispatcher)
        assert my_func.__doc__ == first_doc

    def test_backend_param_docs_merged_into_docstring(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            """Run my_func.

            Parameters
            ----------
            x
                Input value.
            n_jobs
                CPU-only knob.

            Returns
            -------
            Whatever.
            """
            return x

        _trigger_merge(dispatcher)

        doc = my_func.__doc__
        # FakeBackend.my_func docstring describes ``gpu_param`` as
        # "Backend-specific parameter." — that should be lifted in.
        assert "gpu_param" in doc
        assert "Other Parameters" in doc
        assert "gpu_param (fake_gpu)" in doc
        assert "Backend-specific parameter" in doc
        # The backend kwarg gets its own auto-generated entry
        assert "Backend selector injected by ``scverse-backends``" in doc
        # And the merged section sits before Returns
        backend_idx = doc.index("backend")
        other_idx = doc.index("Other Parameters")
        gpu_idx = doc.index("gpu_param")
        returns_idx = doc.index("Returns")
        assert backend_idx < other_idx < gpu_idx < returns_idx

    def test_undocumented_backend_param_still_gets_provenance(self, dispatcher):
        class NoDocBackend:
            name = "no_doc_backend"
            aliases = []

            def my_func(self, x, extra_backend_param=None):
                return x

        dispatcher._registry._backends["no_doc_backend"] = NoDocBackend()
        dispatcher._registry._alias_map["no_doc_backend"] = "no_doc_backend"

        @dispatcher.dispatch
        def my_func(x):
            """Run my_func.

            Parameters
            ----------
            x
                Input value.
            """
            return x

        _trigger_merge(dispatcher)

        doc = my_func.__doc__
        assert "extra_backend_param (no_doc_backend)" in doc

    def test_private_backend_param_is_not_injected(self, dispatcher):
        class PrivateParamBackend:
            name = "private_param_backend"
            aliases = []

            def my_func(self, x, _hidden=None):
                return x

        dispatcher._registry._backends["private_param_backend"] = PrivateParamBackend()
        dispatcher._registry._alias_map["private_param_backend"] = (
            "private_param_backend"
        )

        with pytest.warns(UserWarning, match="must be public"):

            @dispatcher.dispatch
            def my_func(x):
                return x

        sig = inspect.signature(my_func)
        assert "_hidden" not in sig.parameters

    def test_outer_doc_mutation_preserved_on_first_merge(self, dispatcher):
        register_fake(dispatcher)

        def append_doc(fn):
            fn.__doc__ = (fn.__doc__ or "") + "\n\nOuter docs."
            return fn

        @append_doc
        @dispatcher.dispatch
        def my_func(x):
            """Run my_func.

            Parameters
            ----------
            x
                Input value.
            """
            return x

        _trigger_merge(dispatcher)

        assert "Outer docs." in my_func.__doc__
        assert "gpu_param" in my_func.__doc__

    def test_no_backend_no_merge(self, dispatcher):
        """With no backend registered, merge is a no-op beyond `backend=`."""

        @dispatcher.dispatch
        def f(x, n_jobs=None):
            """Plain.

            Parameters
            ----------
            x
                Input.
            """
            return x

        _trigger_merge(dispatcher)
        sig = inspect.signature(f)
        assert "x" in sig.parameters
        assert "backend" in sig.parameters
        # No backend-only params materialize from thin air
        assert "gpu_param" not in sig.parameters
        # Docstring stays unchanged (no extra params injected)
        assert "gpu_param" not in (f.__doc__ or "")

    def test_fallback_function_not_implemented_no_param_added(self, dispatcher):
        """A backend that doesn't implement the host fn shouldn't pollute its signature."""
        register_fake(dispatcher)

        @dispatcher.dispatch
        def some_other_func(y):
            """Other.

            Parameters
            ----------
            y
                Input.
            """
            return y

        _trigger_merge(dispatcher)
        sig = inspect.signature(some_other_func)
        # FakeBackend.my_func exists but FakeBackend.some_other_func does not —
        # so no backend params should be merged.
        assert "gpu_param" not in sig.parameters
        assert "backend" in sig.parameters

    def test_backend_only_param_named_args_is_merged(self, dispatcher):
        class ArgsBackend:
            name = "args_backend"
            aliases = []

            def my_func(self, x, args=None):
                return x

        dispatcher._registry._backends["args_backend"] = ArgsBackend()
        dispatcher._registry._alias_map["args_backend"] = "args_backend"

        @dispatcher.dispatch
        def my_func(x):
            return x

        _trigger_merge(dispatcher)

        sig = inspect.signature(my_func)
        assert "args" in sig.parameters


class TestDiscover:
    def test_discover_triggers_merge(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return x

        # Force a fresh "not yet discovered" state — register_fake bypassed
        # the entrypoint loop, so we need to re-arm discovery to assert that
        # discover() walks through the on_discovered hook.
        dispatcher._registry._discovered = False
        dispatcher.discover()

        assert dispatcher._registry._discovered
        sig = inspect.signature(my_func)
        assert "gpu_param" in sig.parameters

    def test_function_decorated_after_discover_gets_backend_params(self, dispatcher):
        register_fake(dispatcher)
        dispatcher.discover()

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            return x

        sig = inspect.signature(my_func)
        assert "gpu_param" in sig.parameters

    def test_discover_is_idempotent(self, dispatcher):
        register_fake(dispatcher)

        @dispatcher.dispatch
        def my_func(x, n_jobs=None):
            """Run.

            Parameters
            ----------
            x
                Input.
            """
            return x

        dispatcher._registry._discovered = False
        dispatcher.discover()
        first_doc = my_func.__doc__

        # Calling discover again should be safe
        dispatcher.discover()
        dispatcher.discover()
        assert my_func.__doc__ == first_doc
