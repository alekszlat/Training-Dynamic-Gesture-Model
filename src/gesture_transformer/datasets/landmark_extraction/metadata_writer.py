import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetadataRecord:
    """Represents a single record in the metadata manifest."""

    sample_id: str
    source_type: str
    source_name: str
    label: str
    raw_label: str
    path: str
    total_frames: int
    detected_frames: int
    detection_rate: float
    status: str
    landmark_path: str
    error: str


class MetadataWriter:
    """
    Writes metadata records to a CSV file.
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path

    def write(self, records: list[MetadataRecord]):
        fieldnames = [
            "sample_id",
            "source_type",
            "source_name",
            "label",
            "raw_label",
            "path",
            "total_frames",
            "detected_frames",
            "detection_rate",
            "status",
            "landmark_path",
            "error",
        ]

        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = {name: getattr(record, name) for name in fieldnames}
                row["detection_rate"] = f"{record.detection_rate:.2f}"
                writer.writerow(row)
