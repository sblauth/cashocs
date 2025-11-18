# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("."))

import document_cli
import jupytext_process

jupytext_process.process()
document_cli.process()

# -- Project information -----------------------------------------------------

project = "cashocs"
copyright = "2020-2025, Fraunhofer ITWM and Sebastian Blauth"
author = "Sebastian Blauth"

# The full version, including alpha/beta/rc tags
release = "2.7.3"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxarg.ext",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_parser",
    "sphinx_favicon",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True

autodoc_default_options = {
    "member-order": "groupwise",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "alphabetical"
autodoc_mock_imports = [
    "fenics",
    "petsc4py",
    "mpi4py",
    "ufl",
    "ufl_legacy",
    "meshio",
    "dolfin",
    "configparser",
    "h5py",
    "cashocs_extensions",
]
autodoc_typehints = "both"
autoclass_content = "both"

highlight_language = "python"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = ".rst"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

favicons = [
    {"rel": "icon", "sizes": "16x16", "href": "favicon/favicon-16x16.jpg"},
    {"rel": "icon", "sizes": "32x32", "href": "favicon/favicon-32x32.jpg"},
]

suppress_warnings = [
    "autosummary.import_cycle",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

if "dev" in release:
    switcher_version = "dev"
else:
    switcher_version = release

html_theme = "pydata_sphinx_theme"
html_logo = "logo.jpg"
html_theme_options = {
    "github_url": "https://github.com/sblauth/cashocs",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cashocs/",
            "icon": "fa-solid fa-box",
        }
    ],
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "show_nav_level": 1,
    "switcher": {
        "json_url": "https://cashocs.readthedocs.io/en/latest/_static/version_switcher.json",
        "version_match": switcher_version,
    },
    "primary_sidebar_end": [
        "indices.html",
        "sidebar-ethical-ads",
    ],
    "logo": {"text": "cashocs", "alt_text": "cashocs"},
    "navbar_align": "content",
    "show_version_warning_banner": True,
}

html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs", "sidebar-ethical-ads"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["cashocs.css"]

pygments_style = "sphinx"

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False

myst_enable_extensions = ["dollarmath", "colon_fence"]

rst_prolog = """
.. role:: ini(code)
    :language: ini
    :class: highlight

.. role:: python(code)
    :language: python
    :class: highlight
    
.. role:: cpp(code)
    :language: cpp
    :class: highlight

.. role:: bash(code)
    :language: bash
    :class: highlight
"""
