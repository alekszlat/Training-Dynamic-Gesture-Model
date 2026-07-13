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
        with open(self.output_path, 'w') as f:
            # Write header
            f.write("sample_id,source_type,source_name,label,raw_label,path,total_frames,detected_frames,detection_rate,status,landmark_path,error\n")
            for record in records:
                f.write(f"{record.sample_id},{record.source_type},{record.source_name},{record.label},{record.raw_label},{record.path},{record.total_frames},{record.detected_frames},{record.detection_rate:.2f},{record.status},{record.landmark_path},{record.error}\n")