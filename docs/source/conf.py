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
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['templates']

# -- Options for HTML Output -----------------------------------------------------

html_theme = 'sphinx-rst-theme'

# -- Options for EPUB Output -----------------------------------------------------
epub_show_urls = 'footnote'