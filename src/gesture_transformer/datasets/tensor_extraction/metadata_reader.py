import csv

from src.gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)


class MetadataReader:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def _load_metadata(self, status: str = "ok") -> list[MetadataRecord]:
        # Load metadata from the specified path
        metadata = []

        with open(
            self.metadata_path, mode="r", newline="", encoding="utf-8"
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["status"] == status:
                    record = MetadataRecord(
                        sample_id=row["sample_id"],
                        source_type=row["source_type"],
                        source_name=row["source_name"],
                        label=row["label"],
                        raw_label=row["raw_label"],
                        path=row["path"],
                        total_frames=int(row["total_frames"]),
                        detected_frames=int(row["detected_frames"]),
                        detection_rate=float(row["detection_rate"]),
                        status=row["status"],
                        landmark_path=row["landmark_path"],
                        error=row["error"],
                    )
                    metadata.append(record)

        return metadata
