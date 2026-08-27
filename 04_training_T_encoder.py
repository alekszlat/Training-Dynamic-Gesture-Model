from pathlib import Path

import torch
from torch import nn

from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from src.gesture_transformer.training.data_loader import create_data_loader
from src.gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    # Set the model to training mode
    model.train()

    # Initialize variables to track loss and accuracy
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Iterate through the data loader
    for batch in data_loader:
        # Extract features, padding mask, and answers from the batch
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        answers = batch["answers"].to(device)

        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Forward pass through the model
        logits = model(features, padding_mask)

        # Compute the loss using the criterion
        loss = criterion(logits, answers)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Update total loss and accuracy
        total_loss += loss.item() * features.size(0)
        # argmax returns the indices of the maximum values along a specified dimension. In this case, it returns the predicted class labels for each sample in the batch.
        total_correct += (logits.argmax(dim=1) == answers).sum().item()
        total_samples += features.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track loss and accuracy
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        # Iterate through the data loader
        for batch in data_loader:
            # Extract features, padding mask, and answers from the batch
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            answers = batch["answers"].to(device)

            # Forward pass through the model
            logits = model(features, padding_mask)

            # Compute the loss using the criterion
            loss = criterion(logits, answers)

            # Update total loss and accuracy
            total_loss += loss.item() * features.size(0)
            total_correct += (logits.argmax(dim=1) == answers).sum().item()
            total_samples += features.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    checkpoint_dir = Path("src/gesture_transformer/models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"

    # Load the training and validation datasets
    loader = GestureDatasetLoader("data/processed")
    train_data = loader.load("train.pt")
    val_data = loader.load("val.pt")

    # Create GestureDataset instances
    train_dataset = GestureDataset(train_data)
    val_dataset = GestureDataset(val_data)

    # Create DataLoaders for training and validation datasets
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Adjust this based on your system capabilities
    )

    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Adjust this based on your system capabilities
    )

    # Initialize the GestureTransformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureTransformer().to(device)
    print("Selected device:", device)
    print("Model device:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Set up a learning rate scheduler to reduce the learning rate when the validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    num_epochs = 15

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate the model for one epoch
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                best_model_path,
            )

            print(f"Saved new best model (val_loss={val_loss:.4f})")

        current_lr = optimizer.param_groups[0]["lr"]

        # Print the results for the current epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"Learning Rate: {current_lr:.6f}"
        )


if __name__ == "__main__":
    main()
