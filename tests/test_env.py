def test_python_version():
    import sys
    assert sys.version_info.major >= 3, "Python version must be 3 or higher"
    assert sys.version_info.minor >= 10, "Python version must be 3.10 or higher"

def test_torch_import():
    import torch
    assert torch.__version__ >= "2.0.0", "PyTorch version must be 2.0.0 or higher"