"""Unit Tests for GestureDataset and GestureDatasetLoader

Verify all possible error paths and error types they produce,
verify all possible success paths against boundary conditions.

Author: Hristo Hristov
"""

from pathlib import Path

import pytest
import torch

from gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)

NUM_SAMPLES = 2
NUM_FRAMES = 5
NUM_FEATURES = 3
DATA_DIR = "data/processed"


def make_data(**overrides):
    """Build a valid data dictionary for GestureDataset, with optional overrides."""
    data = {
        "x": torch.randn(NUM_SAMPLES, NUM_FRAMES, NUM_FEATURES, dtype=torch.float32),
        "y": torch.randint(0, 2, (NUM_SAMPLES,), dtype=torch.long),
        "mask": torch.ones(NUM_SAMPLES, NUM_FRAMES, dtype=torch.float32),
        "sample_ids": [f"s{i}" for i in range(NUM_SAMPLES)],
        "labels": [f"l{i}" for i in range(NUM_SAMPLES)],
    }
    data.update(overrides)
    return data


def make_fake_path_loader(monkeypatch, *, is_dir=True, is_file=True, torch_load=None):
    """Stub out Path and torch.load so GestureDatasetLoader can be constructed
    and exercised without touching the filesystem."""
    monkeypatch.setattr(Path, "is_dir", lambda self: is_dir)
    monkeypatch.setattr(Path, "is_file", lambda self: is_file)
    if torch_load is not None:
        monkeypatch.setattr(torch, "load", torch_load)
    return GestureDatasetLoader(DATA_DIR)


# ---------------------------------------------------------------------------
# GestureDataset — success paths
# ---------------------------------------------------------------------------


def test_dataset_with_valid_data_creates_dataset():
    # arrange
    data = make_data()

    # act
    dataset = GestureDataset(data)

    # assert
    assert len(dataset) == NUM_SAMPLES


def test_dataset_getitem_returns_expected_sample():
    # arrange
    data = make_data()
    dataset = GestureDataset(data)

    # act
    sample = dataset[0]

    # assert
    assert sample["features"].shape == (NUM_FRAMES, NUM_FEATURES)
    assert sample["answers"].shape == ()
    assert sample["padding_mask"].shape == (NUM_FRAMES,)
    assert sample["sample_id"] == "s0"
    assert sample["label"] == "l0"


def test_dataset_with_zero_samples_creates_empty_dataset():
    # arrange
    data = make_data(
        x=torch.empty(0, NUM_FRAMES, NUM_FEATURES, dtype=torch.float32),
        y=torch.empty(0, dtype=torch.long),
        mask=torch.empty(0, NUM_FRAMES, dtype=torch.float32),
        sample_ids=[],
        labels=[],
    )

    # act
    dataset = GestureDataset(data)

    # assert
    assert len(dataset) == 0


def test_dataset_mask_is_inverted():
    # arrange
    mask = torch.ones(NUM_SAMPLES, NUM_FRAMES)
    mask[0, 0] = 0
    data = make_data(mask=mask)

    # act
    dataset = GestureDataset(data)

    # assert
    assert dataset.padding_mask[0, 0] == True
    assert dataset.padding_mask[0, 1] == False


# ---------------------------------------------------------------------------
# GestureDataset — missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected_message",
    [
        ("x", "Missing feature key 'x'"),
        ("y", "Missing answer key 'y'"),
        ("mask", "Missing padding mask key 'mask'"),
        ("sample_ids", "Missing sample ids key 'sample_ids'"),
        ("labels", "Missing label key 'labels'"),
    ],
    ids=["feature", "answer", "mask", "sample_ids", "labels"],
)
def test_dataset_missing_key_raises_key_error(key, expected_message):
    # arrange
    data = make_data()
    del data[key]

    # act / assert
    with pytest.raises(KeyError, match=expected_message):
        GestureDataset(data)


# ---------------------------------------------------------------------------
# GestureDataset — type errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value,expected_message",
    [
        ("x", "not a tensor", "Features must be a torch.Tensor"),
        ("y", [1, 2], "Answers must be a torch.Tensor"),
        ("mask", "not a tensor", "Padding mask must be a torch.Tensor"),
        ("sample_ids", "not a list", "Sample IDs must be a list"),
        ("labels", "not a list", "Labels must be a list"),
    ],
    ids=["features", "answers", "mask", "sample_ids", "labels"],
)
def test_dataset_invalid_type_raises_type_error(key, value, expected_message):
    # arrange
    data = make_data()
    data[key] = value

    # act / assert
    with pytest.raises(TypeError, match=expected_message):
        GestureDataset(data)


def test_dataset_features_not_float32_raises_type_error():
    # arrange
    data = make_data(
        x=torch.randn(NUM_SAMPLES, NUM_FRAMES, NUM_FEATURES, dtype=torch.float64)
    )

    # act / assert
    with pytest.raises(TypeError, match="Features must use torch.float32"):
        GestureDataset(data)


def test_dataset_answers_not_long_raises_type_error():
    # arrange
    data = make_data(y=torch.randint(0, 2, (NUM_SAMPLES,), dtype=torch.int32))

    # act / assert
    with pytest.raises(TypeError, match="Answers must use torch.long"):
        GestureDataset(data)


# ---------------------------------------------------------------------------
# GestureDataset — shape / dimension errors
# ---------------------------------------------------------------------------


def test_dataset_features_not_3d_raises_value_error():
    # arrange
    data = make_data(x=torch.randn(NUM_SAMPLES, NUM_FRAMES, dtype=torch.float32))

    # act / assert
    with pytest.raises(ValueError, match="Expected features with shape"):
        GestureDataset(data)


def test_dataset_answers_not_1d_raises_value_error():
    # arrange
    data = make_data(y=torch.randint(0, 2, (NUM_SAMPLES, 1), dtype=torch.long))

    # act / assert
    with pytest.raises(ValueError, match="Expected answers with shape"):
        GestureDataset(data)


def test_dataset_features_answers_mismatch_raises_value_error():
    # arrange
    data = make_data(y=torch.randint(0, 2, (NUM_SAMPLES + 1,), dtype=torch.long))

    # act / assert
    with pytest.raises(
        ValueError, match="Number of samples in features and answers must match"
    ):
        GestureDataset(data)


def test_dataset_mask_shape_mismatch_raises_value_error():
    # arrange
    data = make_data(mask=torch.ones(NUM_SAMPLES, NUM_FRAMES + 1, dtype=torch.float32))

    # act / assert
    with pytest.raises(ValueError, match="Padding mask shape must match"):
        GestureDataset(data)


def test_dataset_sample_ids_length_mismatch_raises_value_error():
    # arrange
    data = make_data(sample_ids=["s0"])

    # act / assert
    with pytest.raises(
        ValueError, match="Number of samples in features and sample IDs must match"
    ):
        GestureDataset(data)


def test_dataset_labels_length_mismatch_raises_value_error():
    # arrange
    data = make_data(labels=["l0"])

    # act / assert
    with pytest.raises(
        ValueError, match="Number of samples in features and labels must match"
    ):
        GestureDataset(data)


# ---------------------------------------------------------------------------
# GestureDatasetLoader — constructor
# ---------------------------------------------------------------------------


def test_loader_init_with_non_directory_raises(monkeypatch):
    # arrange
    monkeypatch.setattr(Path, "is_dir", lambda self: False)

    # act / assert
    with pytest.raises(NotADirectoryError, match="Data directory does not exist"):
        GestureDatasetLoader(DATA_DIR)


# ---------------------------------------------------------------------------
# GestureDatasetLoader — load
# ---------------------------------------------------------------------------


def test_loader_load_with_non_existent_file_raises(monkeypatch):
    # arrange
    loader = make_fake_path_loader(monkeypatch, is_file=False)

    # act / assert
    with pytest.raises(FileNotFoundError, match="Data file does not exist"):
        loader.load("missing.pt")


def test_loader_load_with_non_dict_raises(monkeypatch):
    # arrange
    loader = make_fake_path_loader(
        monkeypatch, torch_load=lambda *args, **kwargs: "not a dict"
    )

    # act / assert
    with pytest.raises(TypeError, match="Expected missing.pt to contain a dictionary"):
        loader.load("missing.pt")


def test_loader_load_with_empty_dict_raises(monkeypatch):
    # arrange
    loader = make_fake_path_loader(monkeypatch, torch_load=lambda *args, **kwargs: {})

    # act / assert
    with pytest.raises(ValueError, match="Data file is empty"):
        loader.load("empty.pt")


def test_loader_load_with_valid_dict_returns_dict(monkeypatch):
    # arrange
    expected = {"x": torch.tensor([1])}
    loader = make_fake_path_loader(
        monkeypatch, torch_load=lambda *args, **kwargs: expected
    )

    # act
    result = loader.load("valid.pt")

    # assert
    assert result == expected


# ---------------------------------------------------------------------------
# GestureDatasetLoader — describe
# ---------------------------------------------------------------------------


def test_loader_describe_prints_tensor_info(monkeypatch, capsys):
    # arrange
    loader = make_fake_path_loader(monkeypatch)
    data = {"x": torch.tensor([1, 2]), "y": "not a tensor"}

    # act
    loader.describe(data)
    captured = capsys.readouterr()

    # assert
    assert "x: shape=(2,), dtype=torch.int64" in captured.out
    assert "y: str" in captured.out
