"""Integration tests for the path from stored tensors to a trained checkpoint.

These tests run the real dataset, data loader, model and trainer together with
nothing stubbed. They exist for the mask polarity seam: the stored tensors use
1 for a valid frame, the model wants True for a padded one, and the dataset is
the only place that inverts it. Flip it and nothing raises, the model just
pools the wrong frames, so no unit test of either side can see it.

Author: Hristo Hristov
"""

import pytest
import torch

from gesture_transformer.training.data_loader import create_data_loader
from gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)
from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from gesture_transformer.training.transformer_en_trainer import Trainer

INPUT_DIM = 6
MAX_LEN = 8
SEQUENCE_LENGTH = 4
NUM_CLASSES = 2
HIDDEN_DIM = 8
NUM_HEADS = 2
NUM_LAYERS = 1
DIM_FEEDFORWARD = 16
SEED = 0
OTHER_SEED = 1
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 0.1
CHECKPOINT_NAME = "best_model.pt"
DATA_FILENAME = "train.pt"
VALID_MASKS = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
]
ALL_PADDED_MASK = [0.0, 0.0, 0.0, 0.0]
ALL_PADDED_INDEX = 2
ERROR_PATTERN = "at least one valid frame"


def make_payload(masks=VALID_MASKS):
    generator = torch.Generator().manual_seed(SEED)
    num_samples = len(masks)
    return {
        "x": torch.randn(num_samples, SEQUENCE_LENGTH, INPUT_DIM, generator=generator),
        "y": torch.arange(num_samples) % NUM_CLASSES,
        "mask": torch.tensor(masks, dtype=torch.float32),
        "sample_ids": [f"sample_{index}" for index in range(num_samples)],
        "labels": [f"label_{index % NUM_CLASSES}" for index in range(num_samples)],
    }


def make_model():
    return GestureTransformer(
        input_dim=INPUT_DIM,
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.0,
    )


def make_trainer(tmp_path, masks=VALID_MASKS):
    torch.save(make_payload(masks), tmp_path / DATA_FILENAME)
    data = GestureDatasetLoader(tmp_path).load(DATA_FILENAME)
    loader = create_data_loader(
        GestureDataset(data), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    torch.manual_seed(SEED)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    return Trainer(
        device=torch.device("cpu"),
        save_dir=tmp_path,
        best_model_name_save=CHECKPOINT_NAME,
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
    )


# ---------------------------------------------------------------------------
# Dataset to checkpoint
# ---------------------------------------------------------------------------


def test_dataset_to_trained_checkpoint_round_trip(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path)
    batch = next(iter(trainer.val_loader))
    torch.manual_seed(OTHER_SEED)
    fresh_model = make_model()

    # act
    history = trainer.train(NUM_EPOCHS)
    checkpoint = torch.load(tmp_path / CHECKPOINT_NAME)
    fresh_model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()
    fresh_model.eval()
    with torch.no_grad():
        trained_logits = trainer.model(batch["features"], batch["padding_mask"])
        reloaded_logits = fresh_model(batch["features"], batch["padding_mask"])

    # assert
    assert torch.equal(batch["padding_mask"], torch.tensor(VALID_MASKS) == 0)
    assert history["epoch"] == [1, 2]
    assert checkpoint["epoch"] == NUM_EPOCHS
    assert torch.equal(trained_logits, reloaded_logits)


def test_all_padded_sample_reaches_the_model_and_fails_the_run(tmp_path):
    # arrange
    masks = list(VALID_MASKS)
    masks[ALL_PADDED_INDEX] = ALL_PADDED_MASK
    trainer = make_trainer(tmp_path, masks)

    # act
    with pytest.raises(ValueError, match=ERROR_PATTERN):
        trainer.train(1)

    # assert
    assert not (tmp_path / CHECKPOINT_NAME).exists()
