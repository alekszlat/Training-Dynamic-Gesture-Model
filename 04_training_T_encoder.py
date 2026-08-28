from pathlib import Path

import torch
from torch import nn

from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from src.gesture_transformer.training.data_loader import create_data_loader
from src.gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)
from src.gesture_transformer.training.transformer_en_trainer import Trainer

# Path and filename for saving the best model
SAVE_DIR = Path("src/gesture_transformer/models/checkpoints")
BEST_MODEL_NAME_SAVE = "best_model.pth"

# Device for computaion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gesture dataset loader config
PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = "train.pt"
VAL_FILE = "val.pt"

# DataLoader config
NUM_WORKERS = 0


# Trainer config
NUM_EPOCHS = 15

# Optimizer config
LEARNING_RATE = 5e-4

# Schedular config
MODE = "min"
FACTOR = 5e-1
PATIENCE = 2
MIN_LEARNING_RATE = 1e-6


def main():

    # Initialize the GestureTransformer model
    model = GestureTransformer().to(DEVICE)
    print("Selected device:", DEVICE)
    print("Model device:", next(model.parameters()).device)

    # Load the training and validation datasets
    loader = GestureDatasetLoader(PROCESSED_DIR)
    train_data = loader.load(TRAIN_FILE)
    val_data = loader.load(VAL_FILE)

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
        optimizer, mode=MODE, factor=FACTOR, patience=PATIENCE, min_lr=MIN_LEARNING_RATE
    )

    trainer = Trainer(
        device=DEVICE,
        save_dir=SAVE_DIR,
        best_model_name_save=BEST_MODEL_NAME_SAVE,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.train(num_epochs=15)


if __name__ == "__main__":
    main()
