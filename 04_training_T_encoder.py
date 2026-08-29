import json

import torch
from torch import nn

from config import (
    BEST_MODEL_FILENAME,
    CHECKPOINT_DIR,
    LABEL_MAPPING_FILE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_MODE,
    SCHEDULER_PATIENCE,
    TRAIN_FILE,
    TRAINING_DEVICE,
    VAL_FILE,
)
from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from src.gesture_transformer.training.data_loader import create_data_loader
from src.gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)
from src.gesture_transformer.training.transformer_en_trainer import Trainer


def main():
    # Load the training and validation datasets
    loader = GestureDatasetLoader(PROCESSED_DIR)
    train_data = loader.load(TRAIN_FILE)
    val_data = loader.load(VAL_FILE)

    input_dim = train_data["x"].shape[-1]
    sequence_length = train_data["x"].shape[-2]

    with LABEL_MAPPING_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        label_to_index = json.load(file)

    num_classes = len(label_to_index)

    # Check loaded data
    train_x = train_data["x"]
    val_x = val_data["x"]

    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError(
            "Train and validation feature shapes "
            "do not match: "
            f"{train_x.shape[1:]} != "
            f"{val_x.shape[1:]}"
        )

    num_classes = len(label_to_index)

    expected_indices = set(range(num_classes))
    actual_indices = set(label_to_index.values())

    if actual_indices != expected_indices:
        raise ValueError(
            "Label mapping indices must be contiguous from 0 to num_classes - 1."
        )

    if train_data["y"].max().item() >= num_classes:
        raise ValueError("Training target exceeds model class count.")

    # Initialize the GestureTransformer model
    model = GestureTransformer(
        input_dim=input_dim,
        max_len=sequence_length,
        num_classes=num_classes,
    ).to(TRAINING_DEVICE)
    print("Selected device:", TRAINING_DEVICE)
    print("Model device:", next(model.parameters()).device)

    # Create GestureDataset instances
    train_dataset = GestureDataset(train_data)
    val_dataset = GestureDataset(val_data)

    # Create DataLoaders for training and validation datasets
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,  # Adjust this based on your system capabilities
    )

    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,  # Adjust this based on your system capabilities
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Set up a learning rate scheduler to reduce the learning rate when the validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    trainer = Trainer(
        device=TRAINING_DEVICE,
        save_dir=CHECKPOINT_DIR,
        best_model_name_save=BEST_MODEL_FILENAME,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.train(NUM_EPOCHS)


if __name__ == "__main__":
    main()
