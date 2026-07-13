from dataclasses import dataclass
from pathlib import Path
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
        with open(self.manifest_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 7:
                    continue  # Skip invalid rows
                sample_id, source_type, source_name, external_id, label, raw_label, path_str = parts
                path = Path(path_str)
                record = SampleRecord(sample_id, source_type, source_name, external_id, label, raw_label, path)
                records.append(record)
        return records