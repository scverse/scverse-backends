"""Tests for numpydoc parameter extraction and injection."""

from __future__ import annotations

from scverse_backends._dispatch import _extract_param_docs, _inject_param_docs


class TestDocstringMerging:
    def test_extract_param_docs(self):
        docstring = """\
Run something.

Parameters
----------
x
    Input value.
gpu_param
    Backend-specific parameter.

Returns
-------
Result.
"""
        result = _extract_param_docs(docstring, {"gpu_param"})
        assert "gpu_param" in result
        assert "Backend-specific" in result["gpu_param"]

    def test_extract_skips_missing_params(self):
        docstring = """\
Parameters
----------
x
    Input.
"""
        result = _extract_param_docs(docstring, {"nonexistent"})
        assert result == {}

    def test_extract_no_params_section(self):
        result = _extract_param_docs("Just a docstring.", {"x"})
        assert result == {}

    def test_inject_param_docs(self):
        docstring = """\
Do something.

Parameters
----------
x
    Input.

Returns
-------
Result.
"""
        result = _inject_param_docs(
            docstring,
            {"gpu_param": "gpu_param\n    A backend param."},
            {"gpu_param": {"fake_gpu"}},
        )
        assert "gpu_param (fake_gpu)" in result
        assert "provided by backend" not in result
        assert "Backend selector injected by ``scverse-backends``" in result
        assert "Other Parameters" in result
        # backend doc is host-level; backend-specific docs are separated below.
        lines = result.split("\n")
        other_idx = next(
            i for i, l in enumerate(lines) if l.strip() == "Other Parameters"
        )
        backend_idx = next(i for i, l in enumerate(lines) if l.strip() == "backend")
        gpu_idx = next(
            i for i, l in enumerate(lines) if l.strip() == "gpu_param (fake_gpu)"
        )
        returns_idx = next(i for i, l in enumerate(lines) if l.strip() == "Returns")
        assert backend_idx < other_idx < gpu_idx < returns_idx

    def test_inject_appends_to_existing_other_parameters(self):
        docstring = """\
Do something.

Parameters
----------
x
    Input.

Other Parameters
----------------
rare
    Existing rare parameter.

Returns
-------
Result.
"""
        result = _inject_param_docs(
            docstring,
            {"gpu_param": "gpu_param\n    A backend param."},
            {"gpu_param": {"fake_gpu"}},
        )
        lines = result.split("\n")
        backend_idx = next(i for i, l in enumerate(lines) if l.strip() == "backend")
        other_idx = next(
            i for i, l in enumerate(lines) if l.strip() == "Other Parameters"
        )
        rare_idx = next(i for i, l in enumerate(lines) if l.strip() == "rare")
        gpu_idx = next(
            i for i, l in enumerate(lines) if l.strip() == "gpu_param (fake_gpu)"
        )
        returns_idx = next(i for i, l in enumerate(lines) if l.strip() == "Returns")
        assert backend_idx < other_idx < rare_idx < gpu_idx < returns_idx

    def test_inject_backend_source_preserves_type(self):
        docstring = """\
Do something.

Parameters
----------
x
    Input.
"""
        result = _inject_param_docs(
            docstring,
            {"gpu_param": "gpu_param : bool\n    A backend param."},
            {"gpu_param": {"fake_gpu"}},
        )
        assert "gpu_param (fake_gpu) : bool" in result

    def test_inject_no_params_section_unchanged(self):
        docstring = "Just a plain docstring."
        assert _inject_param_docs(docstring, {"x": "x\n    Param."}) == docstring

    def test_extract_handles_multiline_descriptions(self):
        docstring = """\
Parameters
----------
multi
    First line of description.
    Second line continues here
    with more detail.
other
    Another param.
"""
        result = _extract_param_docs(docstring, {"multi"})
        assert "multi" in result
        assert "Second line" in result["multi"]
        assert "with more detail" in result["multi"]
