# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Ensure local package is importable so autodoc can find modules
sys.path.insert(0, os.path.abspath('..'))

# Mock heavy optional dependencies so autodoc does not import them
autodoc_mock_imports = [
    "numpy",
    "pydicom",
    "SimpleITK",
    "PyQt5",
    "cv2",
    "pandas",
    "scipy",
    "skimage",
    "sklearn",
    "pywt",
    "openpyxl",
    "joblib",
    "tqdm",
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Z-Rad'
copyright = '2025, USZ Medical Physics'
author = 'USZ Medical Physics'
release = '24.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#005f73",
        "color-brand-content": "#0a9396",
    },
}
