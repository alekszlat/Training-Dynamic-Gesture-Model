"""
Turns a session's usable takes into pipeline artifacts: landmark files, a manifest
row and a metadata row, all in the formats the batch pipeline already reads.

Dependencies:
    gesture_transformer.datasets: label formatting, landmark saving, metadata format.

Author:
    Hristo Hristov
"""

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

import numpy as np

from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from gesture_transformer.recording.config import Config


@dataclass(frozen=True)
class TakeRecord:
    """One recorded take, its landmarks and how it scored."""

    path: Path
    label: str
    landmarks: np.ndarray
    total_frames: int
    detected_frames: int
    detection_rate: float
    is_valid: bool


class MetadataAppender:
    """Appends a session's usable takes to the manifest and the landmark metadata."""

    MANIFEST_FIELDNAMES: ClassVar[list[str]] = [
        "sample_id",
        "source_type",
        "source_name",
        "external_id",
        "label",
        "raw_label",
        "path",
    ]

    # Taken from MetadataRecord so the columns cannot drift from the pipeline's.
    METADATA_FIELDNAMES: ClassVar[list[str]] = [
        field.name for field in fields(MetadataRecord)
    ]

    def __init__(self, config: Config, landmark_saver: LandmarkSaver):
        """
        Store the settings and the saver used to write landmark files.

        Args:
            config: Recording session settings.
            landmark_saver: Saver for the extracted landmark arrays.
        """

        self.config = config
        self.landmark_saver = landmark_saver
        self.label_mapper = LabelMapper()

    def append(self, records: list[TakeRecord]) -> None:
        """
        Save the landmarks of every usable take and append its rows.

        Rejected takes are left out, so both files only list samples worth training on.

        Args:
            records: Takes recorded during the session.

        Returns:
            None.
        """

        usable = [record for record in records if record.is_valid]

        if not usable:
            return

        manifest_rows: list[dict[str, str]] = []
        metadata_rows: list[dict[str, str]] = []

        for record in usable:
            sample_id = self._build_sample_id(record.path)
            landmark_path = self.landmark_saver.save(
                sample_id=sample_id,
                landmarks=record.landmarks,
            )

            manifest_rows.append(self._build_manifest_row(record, sample_id))
            metadata_rows.append(
                self._build_metadata_row(record, sample_id, landmark_path)
            )

        self._append_rows(
            self.config.manifest_path, self.MANIFEST_FIELDNAMES, manifest_rows
        )
        self._append_rows(
            self.config.metadata_path, self.METADATA_FIELDNAMES, metadata_rows
        )

        print(f"Appended {len(usable)} takes to {self.config.manifest_path}.")
        print(f"Appended {len(usable)} takes to {self.config.metadata_path}.")
        print(f"Saved {len(usable)} landmark files to {self.config.landmarks_dir}.")

    def _build_sample_id(self, take_path: Path) -> str:
        """
        Build an id from the take's filename, which already carries a timestamp.

        Counting from the local file would restart at one on every machine, so two
        contributors would produce the same ids and overwrite each other's landmark
        files on merge. The timestamp is unique without anyone coordinating.

        Args:
            take_path: Path to the recorded mp4.

        Returns:
            Id for this take.
        """

        return f"recorded_{take_path.stem}"

    def _build_manifest_row(self, record: TakeRecord, sample_id: str) -> dict[str, str]:
        """
        Turn one take into a manifest row.

        Args:
            record: Take to write.
            sample_id: Id this take was given.

        Returns:
            One row keyed by manifest column name.
        """

        return {
            "sample_id": sample_id,
            "source_type": "video",
            "source_name": "recorded",
            "external_id": "",
            "label": self.label_mapper.converter_label(record.label),
            "raw_label": record.label,
            "path": record.path.as_posix(),
        }

    def _build_metadata_row(
        self,
        record: TakeRecord,
        sample_id: str,
        landmark_path: Path,
    ) -> dict[str, str]:
        """
        Turn one take into a landmark metadata row.

        Args:
            record: Take to write.
            sample_id: Id this take was given.
            landmark_path: Path the take's landmarks were saved to.

        Returns:
            One row keyed by metadata column name.
        """

        metadata_record = MetadataRecord(
            sample_id=sample_id,
            source_type="video",
            source_name="recorded",
            label=self.label_mapper.converter_label(record.label),
            raw_label=record.label,
            path=record.path.as_posix(),
            total_frames=record.total_frames,
            detected_frames=record.detected_frames,
            detection_rate=record.detection_rate,
            status="ok",
            landmark_path=str(landmark_path),
            error="",
        )

        return {
            name: getattr(metadata_record, name) for name in self.METADATA_FIELDNAMES
        }

    def _append_rows(
        self,
        output_path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        """
        Append rows to a csv, writing the header only when the file is new.

        Args:
            output_path: File to append to.
            fieldnames: Column names, in order.
            rows: Rows to append.

        Returns:
            None.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not output_path.exists()

        with open(output_path, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if is_new_file:
                writer.writeheader()

            writer.writerows(rows)
