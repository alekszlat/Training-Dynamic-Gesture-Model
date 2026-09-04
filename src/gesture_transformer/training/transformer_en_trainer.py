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
    ) -> tuple[float, float]:

        # Set the model to training mode
        self.model.train()

        # Initialize variables to track loss and accuracy
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Iterate through the data loader
        for batch in self.train_loader:
            # Extract features, padding mask, and answers from the batch
            features = batch["features"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            answers = batch["answers"].to(self.device)

            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()

            # Forward pass through the model
            logits = self.model(features, padding_mask)

            # Compute the loss using the criterion
            loss = self.criterion(logits, answers)

            # Backward pass and optimization step
            loss.backward()
            self.optimizer.step()

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
    ) -> tuple[float, float]:
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize variables to track loss and accuracy
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            # Iterate through the data loader
            for batch in self.val_loader:
                # Extract features, padding mask, and answers from the batch
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                answers = batch["answers"].to(self.device)

                # Forward pass through the model
                logits = self.model(features, padding_mask)

                # Compute the loss using the criterion
                loss = self.criterion(logits, answers)

                # Update total loss and accuracy
                total_loss += loss.item() * features.size(0)
                total_correct += (logits.argmax(dim=1) == answers).sum().item()
                total_samples += features.size(0)

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def train(self, num_epochs: int) -> dict[str, list]:

        history = dict[str, list](
            {
                "epoch": [],
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "learning_rate": [],
            }
        )

        checkpoint_dir = self.save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / self.best_model_name_save

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Train the model for one epoch
            train_loss, train_accuracy = self.train_one_epoch()

            # Validate the model for one epoch
            val_loss, val_accuracy = self.validate_one_epoch()

            # Step the learning rate scheduler
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["learning_rate"].append(current_lr)

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

        return history
