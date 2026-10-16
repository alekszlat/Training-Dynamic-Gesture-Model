"""Unit tests for the gesture transformer trainer.

Covers the training and validation epochs, the scheduler contract, the device
handling and the best-model checkpointing. The model is built tiny with dropout
off and the loaders are unshuffled, so every epoch is deterministic and fast.

A scheduler stepped on the wrong loss, a validation pass that tracks gradients,
or a checkpoint overwritten by a worse epoch all keep training running without
an error, so each of those is pinned by a test that fails when it regresses.

Author: Hristo Hristov
"""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from gesture_transformer.training.transformer_en_trainer import Trainer

INPUT_DIM = 6
MAX_LEN = 8
SEQUENCE_LENGTH = 4
NUM_CLASSES = 3
HIDDEN_DIM = 8
NUM_HEADS = 2
NUM_LAYERS = 1
DIM_FEEDFORWARD = 16
SEED = 0
OTHER_SEED = 1
NUM_SAMPLES = 4
BATCH_SIZE = 2
BATCHES_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE
NUM_EPOCHS = 2
LEARNING_RATE = 0.1
LR_FACTOR = 0.5
CHECKPOINT_NAME = "best_model.pt"
TRAIN_LOSS = 5.0
IMPROVING_VAL_LOSSES = [3.0, 2.0, 1.0]
STALLING_VAL_LOSSES = [1.0, 1.0, 2.0]
WORSENING_VAL_LOSSES = [1.0, 2.0]
BEST_VAL_LOSS = 1.0
HISTORY_KEYS = {
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
}
CHECKPOINT_KEYS = {
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "val_loss",
    "val_accuracy",
}


class TinyGestureDataset(Dataset):
    def __init__(self, num_samples):
        generator = torch.Generator().manual_seed(SEED)
        self.features = torch.randn(
            num_samples, SEQUENCE_LENGTH, INPUT_DIM, generator=generator
        )
        self.padding_mask = torch.zeros(num_samples, SEQUENCE_LENGTH, dtype=torch.bool)
        self.answers = torch.arange(num_samples) % NUM_CLASSES

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "padding_mask": self.padding_mask[index],
            "answers": self.answers[index],
            "sample_id": f"sample_{index}",
            "label": f"label_{self.answers[index].item()}",
        }


class ScriptedCriterion(nn.Module):
    def __init__(self, val_losses):
        super().__init__()
        self.val_losses = val_losses
        self.calls = 0

    def forward(self, logits, answers):
        epoch, position = divmod(self.calls, 2 * BATCHES_PER_EPOCH)
        self.calls += 1
        value = TRAIN_LOSS if position < BATCHES_PER_EPOCH else self.val_losses[epoch]
        return logits.sum() * 0 + value


class SpyScheduler:
    def __init__(self):
        self.steps = []

    def step(self, metric):
        self.steps.append(metric)

    def state_dict(self):
        return {"steps": list(self.steps)}


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, NUM_CLASSES)
        self.feature_devices = []
        self.padding_mask_devices = []
        self.grad_enabled_flags = []

    def forward(self, features, padding_mask):
        self.feature_devices.append(features.device)
        self.padding_mask_devices.append(padding_mask.device)
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        return self.linear(features.mean(dim=1))


class DeviceRecordingCriterion(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.answer_devices = []

    def forward(self, logits, answers):
        self.answer_devices.append(answers.device)
        return super().forward(logits, answers)


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
    return GestureTransformer(**config)


def make_loader(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE):
    return DataLoader(
        TinyGestureDataset(num_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def make_trainer(tmp_path, **overrides):
    torch.manual_seed(SEED)
    model = overrides.pop("model", None) or make_model()
    config = {
        "device": torch.device("cpu"),
        "save_dir": tmp_path,
        "best_model_name_save": CHECKPOINT_NAME,
        "model": model,
        "train_loader": make_loader(),
        "val_loader": make_loader(),
        "optimizer": torch.optim.SGD(model.parameters(), lr=LEARNING_RATE),
        "criterion": nn.CrossEntropyLoss(),
        "scheduler": None,
    }
    config.update(overrides)
    if config["scheduler"] is None:
        config["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            config["optimizer"], mode="min", factor=LR_FACTOR
        )
    return Trainer(**config)


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------


def test_train_one_epoch_updates_the_model_weights(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path)
    weights_before = next(trainer.model.parameters()).clone()

    # act
    trainer.train_one_epoch()

    # assert
    assert not torch.equal(next(trainer.model.parameters()), weights_before)


def test_validate_one_epoch_leaves_the_weights_unchanged(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path)
    weights_before = [parameter.clone() for parameter in trainer.model.parameters()]

    # act
    trainer.validate_one_epoch()

    # assert
    weights_after = list(trainer.model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(weights_before, weights_after)
    )


def test_validate_one_epoch_runs_without_gradient_tracking(tmp_path):
    # arrange
    model = RecordingModel()
    trainer = make_trainer(tmp_path, model=model)

    # act
    trainer.validate_one_epoch()

    # assert
    assert model.grad_enabled_flags == [False] * BATCHES_PER_EPOCH


def test_train_returns_one_history_entry_per_epoch(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path)

    # act
    history = trainer.train(NUM_EPOCHS)

    # assert
    assert set(history) == HISTORY_KEYS
    assert all(len(values) == NUM_EPOCHS for values in history.values())
    assert history["epoch"] == [1, 2]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_steps_once_per_epoch_on_validation_loss(tmp_path):
    # arrange
    spy = SpyScheduler()
    trainer = make_trainer(tmp_path, scheduler=spy)

    # act
    history = trainer.train(NUM_EPOCHS)

    # assert
    assert len(spy.steps) == NUM_EPOCHS
    assert spy.steps == history["val_loss"]


def test_learning_rate_is_unchanged_while_validation_loss_improves(tmp_path):
    # arrange
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=0
    )
    trainer = make_trainer(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=ScriptedCriterion(IMPROVING_VAL_LOSSES),
    )

    # act
    history = trainer.train(len(IMPROVING_VAL_LOSSES))

    # assert
    assert history["learning_rate"] == [LEARNING_RATE] * len(IMPROVING_VAL_LOSSES)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_batches_are_moved_to_the_configured_device(tmp_path):
    # arrange
    device = torch.device("cpu")
    model = RecordingModel()
    criterion = DeviceRecordingCriterion()
    trainer = make_trainer(tmp_path, device=device, model=model, criterion=criterion)

    # act
    trainer.train_one_epoch()

    # assert
    assert model.feature_devices == [device] * BATCHES_PER_EPOCH
    assert model.padding_mask_devices == [device] * BATCHES_PER_EPOCH
    assert criterion.answer_devices == [device] * BATCHES_PER_EPOCH


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_trainer_trains_on_cuda(tmp_path):
    # arrange
    device = torch.device("cuda")
    model = make_model().to(device)
    trainer = make_trainer(tmp_path, device=device, model=model)

    # act
    history = trainer.train(1)

    # assert
    assert next(model.parameters()).device.type == "cuda"
    assert math.isfinite(history["train_loss"][0])


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_best_checkpoint_reloads_into_a_fresh_model(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path)
    trainer.train(1)
    batch = next(iter(trainer.val_loader))
    torch.manual_seed(OTHER_SEED)
    fresh_model = make_model()

    # act
    checkpoint = torch.load(tmp_path / CHECKPOINT_NAME)
    fresh_model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()
    fresh_model.eval()
    with torch.no_grad():
        trained_logits = trainer.model(batch["features"], batch["padding_mask"])
        reloaded_logits = fresh_model(batch["features"], batch["padding_mask"])

    # assert
    assert set(checkpoint) == CHECKPOINT_KEYS
    assert torch.equal(trained_logits, reloaded_logits)


def test_checkpoint_is_not_rewritten_when_validation_loss_worsens(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path, criterion=ScriptedCriterion(STALLING_VAL_LOSSES))

    # act
    trainer.train(len(STALLING_VAL_LOSSES))

    # assert
    checkpoint = torch.load(tmp_path / CHECKPOINT_NAME)
    assert checkpoint["epoch"] == 1


# ---------------------------------------------------------------------------
# Known bugs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.xfail(
    reason="Trainer never calls model.to(device), so forward hits a device mismatch",
    strict=True,
)
def test_trainer_moves_the_model_to_its_configured_device(tmp_path):
    # arrange
    device = torch.device("cuda")
    trainer = make_trainer(tmp_path, device=device, model=make_model())

    # act
    history = trainer.train(1)

    # assert
    assert math.isfinite(history["train_loss"][0])


@pytest.mark.xfail(
    reason="best_val_loss resets to infinity inside train(), so a second call overwrites",
    strict=True,
)
def test_second_train_call_keeps_the_better_checkpoint(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path, criterion=ScriptedCriterion(WORSENING_VAL_LOSSES))
    trainer.train(1)

    # act
    trainer.train(1)

    # assert
    checkpoint = torch.load(tmp_path / CHECKPOINT_NAME)
    assert checkpoint["val_loss"] == BEST_VAL_LOSS


@pytest.mark.xfail(
    reason="the learning rate is read after scheduler.step, so it logs the reduced rate",
    strict=True,
)
def test_logged_learning_rate_is_the_rate_the_epoch_trained_with(tmp_path):
    # arrange
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=0
    )
    trainer = make_trainer(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=ScriptedCriterion(WORSENING_VAL_LOSSES),
    )

    # act
    history = trainer.train(len(WORSENING_VAL_LOSSES))

    # assert
    assert history["learning_rate"] == [LEARNING_RATE] * len(WORSENING_VAL_LOSSES)


@pytest.mark.xfail(
    reason="an empty loader divides by zero samples and raises a bare ZeroDivisionError",
    strict=True,
)
def test_empty_loader_reports_an_empty_loader(tmp_path):
    # arrange
    trainer = make_trainer(tmp_path, train_loader=make_loader(num_samples=0))

    # act / assert
    with pytest.raises((ValueError, RuntimeError)):
        trainer.train_one_epoch()
