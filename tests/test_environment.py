"""Smoke test: the environment is installed and the package is importable.

This is deliberately the cheapest test in the suite. Its job is to fail loudly
and early when someone clones the repo and skips `pip install -r requirements.txt`,
rather than letting them discover it halfway through a notebook.
"""

import importlib


def test_core_dependencies_are_importable():
    for module in ("numpy", "pandas", "sklearn", "pyarrow", "matplotlib"):
        assert importlib.import_module(module) is not None


def test_src_package_is_importable():
    import src

    assert src.__version__ == "0.1.0"


def test_src_subpackages_are_importable():
    for sub in ("data", "features", "models", "evaluation", "viz"):
        assert importlib.import_module(f"src.{sub}") is not None
