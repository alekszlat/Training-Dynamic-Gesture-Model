"""Tests for LandmarkSaver.

The recorder saves without a subdir and relies on files landing directly in
landmarks_dir. Its own tests replace the saver with a mock, so this file is
the only place that default is checked against the real class.
"""

from pathlib import Path

import numpy as np

from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)

LANDMARKS = np.zeros((3, 21, 3), dtype=np.float32)


def test_save_without_subdir_writes_directly_in_landmarks_dir(tmp_path: Path):
    # arrange
    saver = LandmarkSaver(tmp_path / "landmarks")

    # act
    saved_path = saver.save(sample_id="sample", landmarks=LANDMARKS)

    # assert
    assert saved_path == tmp_path / "landmarks" / "sample.npy"
    assert np.load(saved_path).shape == LANDMARKS.shape


def test_save_with_subdir_creates_the_folder_and_writes_in_it(tmp_path: Path):
    # arrange
    saver = LandmarkSaver(tmp_path / "landmarks")

    # act
    saved_path = saver.save(sample_id="sample", landmarks=LANDMARKS, subdir="jester")

    # assert
    assert saved_path == tmp_path / "landmarks" / "jester" / "sample.npy"
    assert np.load(saved_path).shape == LANDMARKS.shape
