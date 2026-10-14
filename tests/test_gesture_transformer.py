"""Unit tests for the gesture transformer.

Covers the output shape, the masked mean pooling over valid frames, the NaN
handling and the padding mask contract. The model is built tiny and in eval
mode with dropout off, so every forward pass is deterministic and fast.

Pooling that divides by the padded length, or lets the content of a padded
frame reach the output, silently makes a gesture's prediction depend on how
much padding its batch happened to carry, so the pooling is covered hardest.

Author: Hristo Hristov
"""

import pytest
import torch

from gesture_transformer.training.gesture_transformer_en import GestureTransformer

INPUT_DIM = 6
MAX_LEN = 8
NUM_CLASSES = 3
HIDDEN_DIM = 8
NUM_HEADS = 2
NUM_LAYERS = 1
DIM_FEEDFORWARD = 16
SEED = 0
GARBAGE_VALUE = 1000.0


def make_model(**overrides):
    config = {
        "input_dim": INPUT_DIM,
        "max_len": MAX_LEN,
        "num_classes": NUM_CLASSES,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": 0.0,
    }
    config.update(overrides)
    return GestureTransformer(**config).eval()


def make_batch(valid_counts, sequence_length=4, pad_value=0.0):
    generator = torch.Generator().manual_seed(SEED)
    frames = torch.randn(len(valid_counts), MAX_LEN + 1, INPUT_DIM, generator=generator)
    x = frames[:, :sequence_length].clone()
    padding_mask = torch.zeros(len(valid_counts), sequence_length, dtype=torch.bool)
    for row, valid_count in enumerate(valid_counts):
        padding_mask[row, valid_count:] = True
    x = x.masked_fill(padding_mask.unsqueeze(-1), pad_value)
    return x, padding_mask


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,num_classes",
    [(1, 2), (4, NUM_CLASSES), (3, 7)],
    ids=["single-sample", "default-classes", "many-classes"],
)
def test_forward_returns_one_logit_row_per_sample_and_class(batch_size, num_classes):
    # arrange
    model = make_model(num_classes=num_classes)
    x, padding_mask = make_batch([3] * batch_size)

    # act
    logits = model(x, padding_mask)

    # assert
    assert logits.shape == (batch_size, num_classes)


def test_forward_rejects_a_sequence_longer_than_max_len():
    # arrange
    model = make_model()
    x, padding_mask = make_batch([3], sequence_length=MAX_LEN + 1)

    # act / assert
    with pytest.raises(RuntimeError, match="must match the size of tensor"):
        model(x, padding_mask)


# ---------------------------------------------------------------------------
# Masked mean pooling
# ---------------------------------------------------------------------------


def test_padding_does_not_change_the_pooled_output():
    # arrange
    torch.manual_seed(SEED)
    model = make_model()
    lightly_padded_x, lightly_padded_mask = make_batch([3], sequence_length=4)
    heavily_padded_x, heavily_padded_mask = make_batch([3], sequence_length=6)

    # act
    lightly_padded_logits = model(lightly_padded_x, lightly_padded_mask)
    heavily_padded_logits = model(heavily_padded_x, heavily_padded_mask)

    # assert
    assert torch.allclose(lightly_padded_logits, heavily_padded_logits, atol=1e-6)


def test_padded_frame_content_does_not_reach_the_output():
    # arrange
    torch.manual_seed(SEED)
    model = make_model()
    zero_padded_x, padding_mask = make_batch([3], sequence_length=5)
    garbage_padded_x, _ = make_batch([3], sequence_length=5, pad_value=GARBAGE_VALUE)

    # act
    zero_padded_logits = model(zero_padded_x, padding_mask)
    garbage_padded_logits = model(garbage_padded_x, padding_mask)

    # assert
    assert torch.allclose(zero_padded_logits, garbage_padded_logits, atol=1e-6)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_at_a_padded_input_frame_is_caught_as_non_finite_output():
    # arrange
    torch.manual_seed(SEED)
    model = make_model()
    x, padding_mask = make_batch([3], sequence_length=5, pad_value=float("nan"))

    # act / assert
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        model(x, padding_mask)


def test_nan_at_a_valid_frame_raises():
    # arrange
    model = make_model()
    x, padding_mask = make_batch([3], sequence_length=5)
    x[0, 1, 0] = float("nan")

    # act / assert
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        model(x, padding_mask)


# ---------------------------------------------------------------------------
# Mask contract
# ---------------------------------------------------------------------------


def test_all_padded_sample_raises():
    # arrange
    model = make_model()
    x, padding_mask = make_batch([0])

    # act / assert
    with pytest.raises(ValueError, match="at least one valid frame"):
        model(x, padding_mask)


def test_one_all_padded_sample_rejects_the_whole_batch():
    # arrange
    model = make_model()
    x, padding_mask = make_batch([3, 0, 3])

    # act / assert
    with pytest.raises(ValueError, match="at least one valid frame"):
        model(x, padding_mask)


def test_non_bool_mask_raises():
    # arrange
    model = make_model()
    x, padding_mask = make_batch([3, 3])
    float_mask = padding_mask.float()

    # act / assert
    with pytest.raises(TypeError, match="boolean"):
        model(x, float_mask)


def test_mask_shape_mismatch_raises():
    # arrange
    model = make_model()
    x, _ = make_batch([3, 3], sequence_length=4)
    wider_mask = torch.zeros(2, 5, dtype=torch.bool)

    # act / assert
    with pytest.raises(ValueError, match="must have shape"):
        model(x, wider_mask)


# ---------------------------------------------------------------------------
# Known bugs
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "A NaN in a padded input frame still poisons the encoder and forward "
        "raises. The desired behaviour is not yet decided: sanitise the input, "
        "or keep failing loudly."
    ),
    strict=True,
)
def test_nan_at_a_padded_frame_does_not_affect_the_output():
    # arrange
    torch.manual_seed(SEED)
    model = make_model()
    zero_padded_x, padding_mask = make_batch([3], sequence_length=5)
    nan_padded_x, _ = make_batch([3], sequence_length=5, pad_value=float("nan"))

    # act
    zero_padded_logits = model(zero_padded_x, padding_mask)
    nan_padded_logits = model(nan_padded_x, padding_mask)

    # assert
    assert torch.isfinite(nan_padded_logits).all()
    assert torch.allclose(zero_padded_logits, nan_padded_logits, atol=1e-6)
