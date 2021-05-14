# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html






# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys



# Check to see whether sbc can be imported.
try:
    import sbc.version
except:
    print("ERROR: can't import sbc.")
    sys.exit(1)






# -- Project information -----------------------------------------------------

project = 'sbc'
copyright = '2021, Matthew Fitzpatrick'
author = 'Matthew Fitzpatrick'






# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.0'



# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']



# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'



# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# The master toctree document.
master_doc = 'index'



# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The short X.Y version.
version = sbc.__version__
# The full version, including alpha/beta/rc tags.
release = sbc.__version__



# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'



# default options for autodoc
autodoc_default_options = {}
autodoc_member_order = 'bysource'



# Avoid a bunch of warnings when using properties with doc strings in classes.
# see https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False



autosummary_generate = True

# For equation numbering by section.
numfig = True
math_numfig = True
numfig_secnum_depth = 3






# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'default'



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []



# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'




# Custom sidebar templates, maps document names to template names.
html_sidebars = {'**': ['localtoc.html',
                        'relations.html',
                        'searchbox.html',
                        'globaltoc.html'],}



# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'sbcdoc'






# -- Extension configuration -------------------------------------------------

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}



# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc,
                    'sbc.tex',
                    'sbc Documentation',
                    author,
                    'manual'),]



# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc,
              'sbc',
              'sbc Documentation',
              [author],
              1)]



# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [('index',
                      'sbc',
                      'sbc Documentation',
                      author,
                      'sbc',
                      'For simulating open-system dynamics of quantum Ising chains.',
                      'Miscellaneous'),]



# cross links to other sphinx documentations
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'tensornetwork': ('https://tensornetwork.readthedocs.io/en/latest/', None)}



# extlinks
extlinks = {'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:'),
            'doi': ('https://dx.doi.org/%s', 'doi:'),
            'manual': ('https://confluence.dwavesys.com/display/~mfitzpatrick/Simulating+dynamics+of+flux+qubit+chains+with+charge+and+hybrid+flux+noise+using+tensor+networks/%s', '')}
