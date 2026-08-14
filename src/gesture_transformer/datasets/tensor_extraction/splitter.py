from __future__ import annotations

from dataclasses import dataclass
from random import Random

from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)


@dataclass
class SplitResult:
    train_records: list[MetadataRecord]
    val_records: list[MetadataRecord]


class Splitter:
    """Creates a stratified train/validation split."""

    def __init__(
        self,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")

        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def split(self, records: list[MetadataRecord]) -> SplitResult:
        """
        Split records into train and validation sets.

        The split is stratified by label, meaning each label is split separately.

        Args:
            records: List of usable metadata records.

        Returns:
            SplitResult containing train_records and val_records.
        """

        records_by_label = self._group_by_label(records)

        train_records: list[MetadataRecord] = []
        val_records: list[MetadataRecord] = []

        rng = Random(self.random_seed)

        for label, label_records in records_by_label.items():
            shuffled_records = label_records.copy()
            rng.shuffle(shuffled_records)

            val_count = self._calculate_val_count(len(shuffled_records))

            val_label_records = shuffled_records[:val_count]
            train_label_records = shuffled_records[val_count:]

            val_records.extend(val_label_records)
            train_records.extend(train_label_records)

            print(
                f"Label '{label}': "
                f"total={len(shuffled_records)}, "
                f"train={len(train_label_records)}, "
                f"val={len(val_label_records)}"
            )

        rng.shuffle(train_records)
        rng.shuffle(val_records)

        return SplitResult(
            train_records=train_records,
            val_records=val_records,
        )

    def _group_by_label(
        self,
        records: list[MetadataRecord],
    ) -> dict[str, list[MetadataRecord]]:
        """Group metadata records by label."""

        records_by_label: dict[str, list[MetadataRecord]] = {}

        for record in records:
            if record.label not in records_by_label:
                records_by_label[record.label] = []

            records_by_label[record.label].append(record)

        return records_by_label

    def _calculate_val_count(self, label_count: int) -> int:
        """
        Calculate how many samples of one label should go to validation.

        Ensures that labels with enough samples get at least one validation sample.
        """

        if label_count <= 1:
            return 0

        val_count = int(label_count * self.val_ratio)

        return max(1, val_count)
