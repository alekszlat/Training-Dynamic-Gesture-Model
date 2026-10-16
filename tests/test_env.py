"""Environment smoke checks.

These do not test project logic; they fail fast when the interpreter or the
torch build in the active environment is older than the project's stated
requirements, which otherwise shows up much later as a confusing runtime error.

Author: Alexander Zlatanov
"""

import sys

from packaging.version import Version

# Project requirements, not the resolved dependency versions in pyproject.toml.
MINIMUM_PYTHON_VERSION = (3, 10)
MINIMUM_TORCH_VERSION = Version("2.0.0")


def test_python_version_meets_the_declared_minimum():
    # act
    python_version = sys.version_info[:2]

    # assert
    assert python_version >= MINIMUM_PYTHON_VERSION


def test_torch_imports_and_meets_the_declared_minimum():
    # act
    import torch

    # assert
    assert Version(torch.__version__) >= MINIMUM_TORCH_VERSION
