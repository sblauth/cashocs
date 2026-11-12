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
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("."))

import document_cli
import document_demos

generated_api_dir = pathlib.Path("./api/generated")
shutil.rmtree(generated_api_dir, ignore_errors=True)
generated_api_dir.mkdir(parents=True, exist_ok=True)

generated_cli_dir = pathlib.Path("./cli/generated")
shutil.rmtree(generated_cli_dir, ignore_errors=True)
generated_cli_dir.mkdir(parents=True, exist_ok=True)

document_cli.process()
document_demos.process()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cashocs"
copyright = "2020-2026, Fraunhofer ITWM and Sebastian Blauth"
author = "Sebastian Blauth"
release = "2.12.0-dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.typer",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_parser",
    "sphinx_favicon",
]

# -- Options for highlighting ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-highlighting

highlight_language = "python"
pygments_style = "sphinx"

# -- Options for markup ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-markup


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

# -- Options for object signatures -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-object-signatures

maximum_signature_line_length = 3
toc_object_entries_show_parents = "hide"

# -- Options for source files ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-source-files

exclude_patterns = ["_build"]
source_suffix = ".rst"

# -- Options for templating --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-templating

templates_path = ["_templates"]

# -- Options for warning control ---------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-warning-control

suppress_warnings = [
    "autosummary.import_cycle",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
html_static_path = ["_static"]
html_css_files = ["cashocs.css"]

# -- Configuration for sphinx.ext.napoleon -----------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

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

# -- Configuration for sphinx.et.autodoc -------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

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
    "numpy",
    "scipy",
    "matplotlib",
]
autodoc_typehints = "both"
autoclass_content = "both"

# -- Configuration for sphinx_favicon ----------------------------------------
# https://sphinx-favicon.readthedocs.io/en/latest/configuration.html

favicons = [
    {"rel": "icon", "sizes": "16x16", "href": "favicon/favicon-16x16.jpg"},
    {"rel": "icon", "sizes": "32x32", "href": "favicon/favicon-32x32.jpg"},
]

# -- Configuration for sphinx.ext.autosummary --------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False

# -- Configuration for myst_parser -------------------------------------------
# https://myst-parser.readthedocs.io/en/v0.15.1/sphinx/reference.html

myst_enable_extensions = ["dollarmath", "colon_fence"]
