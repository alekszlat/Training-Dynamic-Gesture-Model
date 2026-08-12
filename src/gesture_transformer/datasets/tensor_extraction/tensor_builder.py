from pathlib import Path

import numpy as np
import torch

from src.gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from src.gesture_transformer.datasets.tensor_extraction.feature_builder import (
    FeatureBuilder,
)
from src.gesture_transformer.datasets.tensor_extraction.label_encoder import (
    LabelEncoder,
)
from src.gesture_transformer.datasets.tensor_extraction.metadata_reader import (
    MetadataReader,
)
from src.gesture_transformer.datasets.tensor_extraction.splitter import Splitter
from src.gesture_transformer.datasets.tensor_extraction.temporal_normalizer import (
    TemporalNormalizer,
)
from src.gesture_transformer.datasets.tensor_extraction.tensor_saver import TensorSaver


class TensorBuilder:
    """Builds train.pt and val.pt from extracted landmark files."""

    def __init__(
        self,
        metadata_path: Path,
        processed_dir: Path,
        label_list: list[str],
        target_length: int = 40,
    ) -> None:
        self.metadata_path = metadata_path
        self.processed_dir = processed_dir
        self.label_list = label_list
        self.target_length = target_length

        self.metadata_reader = MetadataReader(metadata_path)
        self.splitter = Splitter(val_ratio=0.2, random_seed=42)
        self.label_encoder = LabelEncoder(label_list)
        self.normalizer = TemporalNormalizer(target_length=target_length)
        self.feature_builder = FeatureBuilder()
        self.tensor_saver = TensorSaver(processed_dir)

    def build(self) -> None:
        """Run the full tensor building pipeline."""

        # 1. Read usable metadata rows
        records = self.metadata_reader._load_metadata()

        print(f"Usable records: {len(records)}")

        # 2. Split records into train and validation
        split_result = self.splitter.split(records)

        print(f"Train records: {len(split_result.train_records)}")
        print(f"Validation records: {len(split_result.val_records)}")

        # 3. Build train tensors
        train_data = self._build_split_tensors(split_result.train_records)

        # 4. Build validation tensors
        val_data = self._build_split_tensors(split_result.val_records)

        # 5. Save tensors
        self.tensor_saver.save_split(
            filename="train.pt",
            data=train_data,
        )

        self.tensor_saver.save_split(
            filename="val.pt",
            data=val_data,
        )

        # 6. Save label mapping
        self.tensor_saver.save_label_mapping(self.label_encoder.mapping())

        print("Tensor building finished.")

    def _build_split_tensors(
        self,
        records: list[MetadataRecord],
    ) -> dict:
        """
        Build x, y, mask, sample_ids, and labels for one split.

        This is used for both train_records and val_records.
        """

        x_list: list[np.ndarray] = []
        y_list: list[int] = []
        mask_list: list[np.ndarray] = []
        sample_ids: list[str] = []
        labels: list[str] = []

        for record in records:
            # 1. Load raw landmarks [T, 21, 3]
            landmarks = np.load(record.landmark_path)

            # 2. Normalize to [40, 21, 3]
            normalized = self.normalizer.normalize(landmarks)

            # 3. Build features [40, 126]
            features = self.feature_builder.build(
                landmarks=normalized.norm_sequence,
                mask=normalized.mask,
            )

            # 4. Encode label string to integer
            label_index = self.label_encoder.encode(record.label)

            # 5. Collect sample data
            x_list.append(features)
            y_list.append(label_index)
            mask_list.append(normalized.mask)
            sample_ids.append(record.sample_id)
            labels.append(record.label)

        # 6. Stack into tensors
        x = torch.tensor(np.stack(x_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.long)
        mask = torch.tensor(np.stack(mask_list), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "sample_ids": sample_ids,
            "labels": labels,
        }
