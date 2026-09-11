"""Tests for TemporalNormalizer.

Padding marked as valid would make FeatureBuilder read the real-to-zero
boundary as one large genuine movement, with nothing later on raising.
These tests verify the length and the mask.

Author: Hristo Hristov
"""

import numpy as np
import pytest

from gesture_transformer.datasets.tensor_extraction.temporal_normalizer import (
    TemporalNormalizer,
)

TARGET_MASK_LENGTH = 40


@pytest.mark.parametrize(
    "input_shape",
    [(60, 21, 3), (33, 21, 3), (40, 21, 3)],
    ids=["more-frames", "less frames", "equal frames"],
)
def test_normalizer_with_all_variants_succeeds(input_shape):
    # arrange
    data = np.random.randint(1, 20, size=(input_shape))

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    assert normalized_data.norm_sequence.shape == (TARGET_MASK_LENGTH, 21, 3)
    assert normalized_data.mask.shape == (TARGET_MASK_LENGTH,)


def test_normalizer_with_less_frames_pads_zeros():
    # arrange
    FRAME_COUNT = 30
    data = np.random.randint(1, 20, size=(FRAME_COUNT, 21, 3))

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    # Unpadded frames remain unchanged
    assert np.array_equal(normalized_data.norm_sequence[:FRAME_COUNT], data)

    # Padding contains only zeros
    assert not np.any(normalized_data.norm_sequence[FRAME_COUNT:TARGET_MASK_LENGTH])


def test_normalizer_with_empty_sequence_returns_zero_matrix():
    # arrange
    data = np.empty((0, 21, 3))

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    assert np.array_equal(
        normalized_data.norm_sequence, np.zeros((TARGET_MASK_LENGTH, 21, 3))
    )


def test_normalizer_with_less_frames_marks_padding_invalid():
    # arrange
    FRAME_COUNT = 30
    data = np.random.randint(1, 20, size=(FRAME_COUNT, 21, 3))

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    expected_mask = np.zeros(TARGET_MASK_LENGTH)
    expected_mask[:FRAME_COUNT] = 1.0

    assert np.array_equal(normalized_data.mask, expected_mask)


@pytest.mark.parametrize(
    "input_shape",
    [(60, 21, 3), (40, 21, 3)],
    ids=["more-frames", "equal-frames"],
)
def test_normalizer_without_padding_marks_every_frame_valid(input_shape):
    # arrange
    data = np.random.randint(1, 20, size=input_shape)

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    assert np.all(normalized_data.mask == 1.0)


def test_normalizer_with_empty_sequence_marks_no_frame_valid():
    # arrange
    data = np.empty((0, 21, 3))

    # act
    normalizer = TemporalNormalizer(TARGET_MASK_LENGTH)
    normalized_data = normalizer.normalize(data)

    # assert
    assert not np.any(normalized_data.mask)
