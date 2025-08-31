import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

import re
from pathlib import Path

init_file = Path(__file__).resolve().parents[2] / "src" / "stelpar" / "__init__.py"

# -- Project Information -----------------------------------------------------

project = 'stelpar'
copyright = '2025, Matt Fields'
author = 'Matt Fields'

version_match = re.search(
    r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
    init_file.read_text(encoding="utf-8"),
    re.MULTILINE
)

if version_match:
    # The full version, including alpha/beta/rc tags
    release = version_match.group(1)
    # The short X.Y version
    version = version_match.group(1)
else:
    raise RuntimeError("Unable to find __version__ in __init__.py")

# -- General Configuration -----------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_nb',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc'
]

# enables '$...$' latex math rendering
myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]

source_suffix = ".rst"
master_doc = "index"

viewcode_follow_imported_members = True
autosummary_ignore_module_all = False
autodoc_mock_imports = [
    'numpy',
    'isochrones',
    'astropy',
    'astroquery',
    'synphot',
    'scipy',
    'dust_extinction',
    'pandas',
    'numba',
    'matplotlib'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ["_templates"]

# -- Options for HTML Output -----------------------------------------------------
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "stelpar"
html_static_path = ["_static"]
html_favicon = "_static/stelpar_favicon.ico"
html_logo = "_static/stelpar_favicon.svg"
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/mjfields/stelpar",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "logo" : {
        "text" : f"stelpar ({version})"
    },
    "show_navbar_depth": 2,
    "collapse_navigation": True,
    "sticky_navigation": True
}
nb_execution_mode = "off"
nb_execution_timeout = -1

# -- Options for EPUB Output -----------------------------------------------------
epub_show_urls = 'footnote'

# -- Plot options ----------------------------------------------------------------
plot_include_source = True
plot_formats = [('png', 100), 'pdf']