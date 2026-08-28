from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        device: torch.device,
        save_dir: Path,
        best_model_name_save: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        self.save_dir = save_dir
        self.best_model_name_save = best_model_name_save
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler

    def train_one_epoch(
        self,
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
            features = batch["features"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            answers = batch["answers"].to(self.device)

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
        self,
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
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                answers = batch["answers"].to(self.device)

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

    def train(self, num_epochs: int):

        checkpoint_dir = self.save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / self.best_model_name_save

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Train the model for one epoch
            train_loss, train_accuracy = self.train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
            )

            # Validate the model for one epoch
            val_loss, val_accuracy = self.validate_one_epoch(
                self.model,
                self.val_loader,
                self.criterion,
            )

            # Step the learning rate scheduler
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                    best_model_path,
                )

                print(f"Saved new best model (val_loss={val_loss:.4f})")

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print the results for the current epoch
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Learning Rate: {current_lr:.6f}"
            )
