import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project Information -----------------------------------------------------

project = 'stelpar'
copyright = '2025, Matt Fields'
author = 'Matt Fields'

# The short X.Y version
version = '0.1.0'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General Configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'IPython.sphinxext.ipython_console_highlighting'
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