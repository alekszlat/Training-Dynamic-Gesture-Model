import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleRecord:
    """Represents a single record in the sample manifest."""

    sample_id: str
    source_type: str
    source_name: str
    external_id: str
    label: str
    raw_label: str
    path: Path


class SampleManifestReader:
    """
    Reads and parses a sample manifest CSV file.
    Main role is to read the manifest and return a list of SampleRecord objects.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path

    def read_manifest(self):
        records = []
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                sample = SampleRecord(
                    sample_id=row["sample_id"],
                    source_type=row["source_type"],
                    source_name=row["source_name"],
                    external_id=row["external_id"],
                    label=row["label"],
                    raw_label=row["raw_label"],
                    path=Path(row["path"]),
                )
                records.append(sample)
        return records
