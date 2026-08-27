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
        features = batch["features"]
        padding_mask = batch["padding_mask"]
        answers = batch["answers"]

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
            features = batch["features"]
            padding_mask = batch["padding_mask"]
            answers = batch["answers"]

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
    model = GestureTransformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    num_epochs = 15

    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        # Validate the model for one epoch
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion)

        # Print the results for the current epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
