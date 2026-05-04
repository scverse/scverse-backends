"""Sphinx configuration for scverse-backends."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import metadata

# -- Project metadata --------------------------------------------------------
info = metadata("scverse-backends")
project = info["Name"]
author = "scverse"
copyright = f"{datetime.now():%Y}, {author}"
release = info["Version"]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "scanpydoc",
]

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
suppress_warnings = ["myst.header"]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

# -- Autodoc -----------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
typehints_defaults = "comma"
always_document_param_types = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "scanpydoc"
html_title = "scverse-backends"
html_static_path = ["_static"]
html_show_sphinx = False
pygments_style = "default"
pygments_dark_style = "native"

html_theme_options = {
    "repository_url": "https://github.com/scverse/scverse-backends",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "navigation_with_keys": False,
}

html_context = {
    "github_user": "scverse",
    "github_repo": "scverse-backends",
    "github_version": "main",
    "doc_path": "docs",
}
