import torch
from torch.utils.data import DataLoader

from gesture_transformer.training.gesture_dataset import GestureDataset


def create_data_loader(
    dataset: GestureDataset, batch_size: int, shuffle: bool, num_workers: int = 0
):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


if __name__ == "__main__":
    from gesture_transformer.training.gesture_dataset import GestureDatasetLoader

    loader = GestureDatasetLoader("data/processed")
    train_data = loader.load("train.pt")
    val_data = loader.load("val.pt")
    train_dataset = GestureDataset(train_data)
    val_dataset = GestureDataset(val_data)

    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Later, you can set this to a higher number for better performance if your system supports it.
    )

    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Later, you can set this to a higher number for better performance if your system supports it.
    )

    batch = next(iter(train_loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value).__name__}")
