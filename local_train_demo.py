"""
Trains a gesture model on the locally extracted landmarks, for the webcam demo.

Separate from the numbered pipeline scripts. It reads its own metadata file and
writes everything under outputs/local_demo/, so a demo run never touches the
dataset the pipeline builds.

Run with:

    uv run python local_train_demo.py

Dependencies:
    gesture_transformer.datasets: tensor building from landmark metadata.
    gesture_transformer.training: dataset, model and training loop.

Author:
    Hristo Hristov
"""

import json
from pathlib import Path

import torch
from torch import nn

from gesture_transformer.datasets.tensor_extraction.tensor_builder import TensorBuilder
from gesture_transformer.training.data_loader import create_data_loader
from gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)
from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from gesture_transformer.training.transformer_en_trainer import Trainer

DEMO_DIR = Path("outputs/local_demo")
METADATA_PATH = DEMO_DIR / "metadata.csv"
PROCESSED_DIR = DEMO_DIR / "processed"

# Only the four swipe classes are present locally, so the label list is
# narrowed here instead of using config.SUPPORTED_LABELS.
LABEL_LIST = ["swiping_down", "swiping_left", "swiping_right", "swiping_up"]

TARGET_SEQUENCE_LENGTH = 40
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tensors() -> None:
    """
    Build train.pt and val.pt from the demo metadata.

    Returns:
        None.
    """

    TensorBuilder(
        metadata_path=METADATA_PATH,
        processed_dir=PROCESSED_DIR,
        label_list=LABEL_LIST,
        target_length=TARGET_SEQUENCE_LENGTH,
    ).build()


def main() -> None:
    """
    Train on the demo tensors and write the checkpoint the webcam demo reads.

    Returns:
        None.
    """

    build_tensors()

    loader = GestureDatasetLoader(PROCESSED_DIR)
    train_data = loader.load("train.pt")
    val_data = loader.load("val.pt")

    input_dim = train_data["x"].shape[-1]
    sequence_length = train_data["x"].shape[-2]

    with (PROCESSED_DIR / "label_to_index.json").open(encoding="utf-8") as file:
        label_to_index = json.load(file)

    num_classes = len(label_to_index)

    counts = torch.bincount(train_data["y"], minlength=num_classes)
    for label, index in label_to_index.items():
        print(f"{label}: {counts[index].item()} training samples")

    model = GestureTransformer(
        input_dim=input_dim,
        max_len=sequence_length,
        num_classes=num_classes,
    ).to(DEVICE)

    train_loader = create_data_loader(
        dataset=GestureDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = create_data_loader(
        dataset=GestureDataset(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    trainer = Trainer(
        device=DEVICE,
        save_dir=DEMO_DIR,
        best_model_name_save="best_model.pth",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    history = trainer.train(NUM_EPOCHS)

    # Trainer saves weights only, so the webcam demo has no way to rebuild the
    # model without these.
    with (DEMO_DIR / "model_args.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input_dim": input_dim,
                "max_len": sequence_length,
                "num_classes": num_classes,
            },
            file,
            indent=4,
        )

    with (DEMO_DIR / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    best = min(history["val_loss"])
    index = history["val_loss"].index(best)
    print(
        f"best epoch {history['epoch'][index]}: "
        f"val_loss {best:.4f}, val_accuracy {history['val_accuracy'][index]:.4f}"
    )


if __name__ == "__main__":
    main()
