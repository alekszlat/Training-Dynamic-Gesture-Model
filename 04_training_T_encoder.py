import torch

from gesture_transformer.training.gesture_transformer_en import GestureTransformer
from src.gesture_transformer.training.data_loader import create_data_loader
from src.gesture_transformer.training.gesture_dataset import (
    GestureDataset,
    GestureDatasetLoader,
)


def main():
    # Load the training and validation datasets
    loader = GestureDatasetLoader("data/processed")
    train_data = loader.load("train.pt")
    # val_data = loader.load("val.pt")

    # Create GestureDataset instances
    train_dataset = GestureDataset(train_data)
    # val_dataset = GestureDataset(val_data)

    # Create DataLoaders for training and validation datasets
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Adjust this based on your system capabilities
    )

    # val_loader = create_data_loader(
    #    dataset=val_dataset,
    #    batch_size=32,
    #    shuffle=False,
    #    num_workers=0,  # Adjust this based on your system capabilities
    # )

    # Example: Iterate through a batch from the training DataLoader
    batch = next(iter(train_loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value).__name__}")

    # Example: Create an instance of the GestureTransformer and pass a batch through it
    features = batch["features"]
    padding_mask = batch["padding_mask"]
    answers = batch["answers"]

    print(f"Feature shape: {tuple(features.shape)}")
    model = GestureTransformer()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()

    optimizer.zero_grad()

    logits = model(
        features,
        padding_mask,
    )

    loss = criterion(
        logits,
        answers,
    )

    loss.backward()

    optimizer.step()

    print("Features:", features.shape)
    print("Logits:", logits.shape)
    print("Answers:", answers.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
