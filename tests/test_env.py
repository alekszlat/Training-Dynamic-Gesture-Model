import sys

from packaging.version import Version


def test_python_version():
    assert sys.version_info >= (3, 10), "Python 3.10 or higher required"


def test_torch_import():
    import torch

    assert Version(torch.__version__) >= Version("2.0.0"), (
        f"PyTorch 2.0.0 or higher required, got {torch.__version__}"
    )
