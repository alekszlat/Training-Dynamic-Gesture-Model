from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import Dataset


class GestureSample(TypedDict):
    features: torch.Tensor
    answers: torch.Tensor
    padding_mask: torch.Tensor
    sample_id: str
    label: str


class GestureDataset(Dataset):
    def __init__(
        self,
        data: dict,
        feature_key: str = "x",
        answer_key: str = "y",
        padding_mask: str = "mask",
        sample_ids: str = "sample_ids",
        label_key: str = "labels",
    ):
        if feature_key not in data:
            raise KeyError(
                f"Missing feature key '{feature_key}'. "
                f"Available keys: {list(data.keys())}"
            )

        if answer_key not in data:
            raise KeyError(
                f"Missing answer key '{answer_key}'. "
                f"Available keys: {list(data.keys())}"
            )

        if padding_mask not in data:
            raise KeyError(
                f"Missing padding mask key '{padding_mask}'. "
                f"Available keys: {list(data.keys())}"
            )

        if sample_ids not in data:
            raise KeyError(
                f"Missing sample ids key '{sample_ids}'. "
                f"Available keys: {list(data.keys())}"
            )

        if label_key not in data:
            raise KeyError(
                f"Missing label key '{label_key}'. Available keys: {list(data.keys())}"
            )

        self.features = data[feature_key]
        self.answers = data[answer_key]
        self.padding_mask = data[padding_mask] == 0
        self.sample_ids = data[sample_ids]
        self.labels = data[label_key]
        expected_mask_shape = self.features.shape[:2]

        if not isinstance(self.features, torch.Tensor):
            raise TypeError(
                f"Features must be a torch.Tensor, got {type(self.features)}"
            )

        if not isinstance(self.answers, torch.Tensor):
            raise TypeError(f"Answers must be a torch.Tensor, got {type(self.answers)}")

        if not isinstance(self.padding_mask, torch.Tensor):
            raise TypeError(
                f"Padding mask must be a torch.Tensor, got {type(self.padding_mask)}"
            )

        if not isinstance(self.sample_ids, list):
            raise TypeError(f"Sample IDs must be a list, got {type(self.sample_ids)}")

        if not isinstance(self.labels, list):
            raise TypeError(f"Labels must be a list, got {type(self.labels)}")

        if self.features.dtype != torch.float32:
            raise TypeError(
                f"Features must use torch.float32. Got {self.features.dtype}."
            )

        if self.answers.dtype != torch.long:
            raise TypeError(
                "Answers must use torch.long (torch.int64) "
                "for CrossEntropyLoss. "
                f"Got {self.answers.dtype}."
            )

        if self.padding_mask.dtype != torch.bool:
            raise TypeError(
                f"Padding mask must use torch.bool. Got {self.padding_mask.dtype}."
            )

        if self.features.dim() != 3:
            raise ValueError(
                "Expected features with shape "
                "[samples, frames, features]. "
                f"Got {tuple(self.features.shape)}."
            )

        if self.answers.dim() != 1:
            raise ValueError(
                "Expected answers with shape [samples]. "
                f"Got {tuple(self.answers.shape)}."
            )

        if self.features.shape[0] != self.answers.shape[0]:
            raise ValueError(
                "Number of samples in features and answers must match. "
                f"Got {self.features.shape[0]} and {self.answers.shape[0]}."
            )

        if self.padding_mask.shape != expected_mask_shape:
            raise ValueError(
                "Padding mask shape must match the sample and frame "
                "dimensions of features. "
                f"Expected {tuple(expected_mask_shape)}, "
                f"got {tuple(self.padding_mask.shape)}."
            )

        if self.features.shape[0] != len(self.sample_ids):
            raise ValueError(
                "Number of samples in features and sample IDs must match. "
                f"Got {self.features.shape[0]} and {len(self.sample_ids)}."
            )

        if self.features.shape[0] != len(self.labels):
            raise ValueError(
                "Number of samples in features and labels must match. "
                f"Got {self.features.shape[0]} and {len(self.labels)}."
            )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> GestureSample:
        return {
            "features": self.features[idx],
            "answers": self.answers[idx],
            "padding_mask": self.padding_mask[idx],
            "sample_id": self.sample_ids[idx],
            "label": self.labels[idx],
        }


class GestureDatasetLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory does not exist: {self.data_dir}")

    def load(self, filename: str) -> dict[str, Any]:
        file_path = self.data_dir / filename

        if not file_path.is_file():
            raise FileNotFoundError(f"Data file does not exist: {file_path}")

        data = torch.load(
            file_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected {filename} to contain a dictionary, "
                f"but got {type(data).__name__}."
            )

        if len(data) == 0:
            raise ValueError(f"Data file is empty: {file_path}")

        return data

    def describe(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value).__name__}")


if __name__ == "__main__":
    loader = GestureDatasetLoader("data/processed")
    train_data = loader.load("train.pt")
    print(type(train_data))
    train_dataset = GestureDataset(train_data)

    print("Dataset length:", len(train_dataset))

    sample = train_dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value).__name__}")

    print(type(train_data["sample_ids"]))
    print(type(train_data["labels"]))
    print(type(train_data["x"]))
    print(type(train_data["y"]))
    print(type(train_data["mask"]))

    print(train_data["x"].shape)
    print(train_data["y"].shape)
    print(train_data["mask"].shape)

    print(torch.unique(train_data["mask"]))
    print(train_data["mask"][0])
    print(train_data["x"][0].abs().sum(dim=-1))
