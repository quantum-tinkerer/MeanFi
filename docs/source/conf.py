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

import meanfi  # noqa: F401

package_path = os.path.abspath("../meanfi")
# Suppress superfluous frozen modules warning.
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
sys.path.insert(0, package_path)


# -- Project information -----------------------------------------------------

project = "MeanFi"
copyright = "2024, MeanFi developers"
author = "MeanFi developers"
gitlab_url = "https://gitlab.kwant-project.org/qt/meanfi"

# The full version, including alpha/beta/rc tags
release = meanfi.__version__
major, minor = meanfi.__version_tuple__[:2]
version = f"{major}.{minor}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx_tippy",
]
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "substitution",
    "colon_fence",
]
nb_execution_timeout = 500
nb_execution_raise_on_error = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "kwant": ("https://kwant-project.org/doc/1", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

default_role = "autolink"

latex_elements = {"extrapackages": r"\usepackage{braket}"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autoclass_content = "both"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# html_title = "$MeanFi$"

html_theme_options = {
    "repository_url": "https://gitlab.kwant-project.org/qt/meanfi",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "repository_branch": "main",
    "use_download_button": True,
    "home_page_in_toc": True,
    "logo": {
        "alt_text": "MeanFi",
        "image_light": "logo_imgs/MeanFi_logo_light.svg",
        "image_dark": "logo_imgs/MeanFi_logo_dark.svg",
    },
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
