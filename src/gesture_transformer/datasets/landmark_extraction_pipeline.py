from pathlib import Path
from typing import Iterable

from src.gesture_transformer.datasets.landmark_extractor import (
    LandmarkExtractor,
    LandmarkExtractionResult,
)
from src.gesture_transformer.datasets.landmark_saver import LandmarkSaver
from src.gesture_transformer.datasets.metadata_writer import (
    MetadataWriter,
    MetadataRecord,
)
from src.gesture_transformer.datasets.manifest.sample_manifest_reader import (
    SampleManifestReader,
)
from src.gesture_transformer.datasets.readers.reader_factory import ReaderFactory
from src.gesture_transformer.datasets.readers.base_reader import Frame


class LandmarkExtractionPipeline:
    """Pipeline for extracting MediaPipe landmarks from gesture samples."""

    def __init__(
        self,
        manifest_reader: SampleManifestReader,
        reader_factory: ReaderFactory,
        landmark_extractor: LandmarkExtractor,
        landmark_saver: LandmarkSaver,
        metadata_writer: MetadataWriter,
        min_detection_rate: float = 0.7,
    ) -> None:
        self.manifest_reader = manifest_reader
        self.reader_factory = reader_factory
        self.landmark_extractor = landmark_extractor
        self.landmark_saver = landmark_saver
        self.metadata_writer = metadata_writer
        self.min_detection_rate = min_detection_rate

    def run(self) -> None:
        """Run the full landmark extraction pipeline."""

        records: list[MetadataRecord] = []

        try:
            samples = self.manifest_reader.read_manifest()

            for index, sample in enumerate(samples, start=1):
                print(f"[{index}/{len(samples)}] Processing {sample.sample_id}")

                try:
                    record = self._process_sample(sample)
                    records.append(record)

                    print(
                        f"  status={record.status}, "
                        f"detected={record.detected_frames}/{record.total_frames}, "
                        f"rate={record.detection_rate}"
                    )

                except Exception as error:
                    records.append(
                        MetadataRecord(
                            sample_id=sample.sample_id,
                            source_type=sample.source_type,
                            source_name=sample.source_name,
                            label=sample.label,
                            raw_label=sample.raw_label,
                            path=str(sample.path),
                            total_frames=0,
                            detected_frames=0,
                            detection_rate=0.0,
                            status="error",
                            landmark_path="",
                            error=str(error),
                        )
                    )

                    print(f"  status=error, error={error}")

            self.metadata_writer.write(records)
            self._print_summary(records)

        finally:
            self.landmark_extractor.close()

    def _process_sample(self, sample) -> MetadataRecord:
        """Process one sample from samples.csv."""

        reader = self.reader_factory.get_reader(sample.source_type)
        frames: Iterable[Frame] = reader.read_frames(sample.path)

        extraction_result: LandmarkExtractionResult = self.landmark_extractor.extract(frames)

        detection_rate = self._calculate_detection_rate(
            total_frames=extraction_result.total_frames,
            detected_frames=extraction_result.detected_frames,
        )

        status, error = self._get_status(extraction_result, detection_rate)

        landmark_path = ""

        if status == "ok":
            saved_path = self.landmark_saver.save(
                sample_id=sample.sample_id,
                landmarks=extraction_result.landmarks,
            )
            landmark_path = str(saved_path)

        return MetadataRecord(
            sample_id=sample.sample_id,
            source_type=sample.source_type,
            source_name=sample.source_name,
            label=sample.label,
            raw_label=sample.raw_label,
            path=str(sample.path),
            total_frames=extraction_result.total_frames,
            detected_frames=extraction_result.detected_frames,
            detection_rate=detection_rate,
            status=status,
            landmark_path=landmark_path,
            error=error,
        )

    def _calculate_detection_rate(
        self,
        total_frames: int,
        detected_frames: int,
    ) -> float:
        """Calculate detected_frames / total_frames safely."""

        if total_frames == 0:
            return 0.0

        return round(detected_frames / total_frames, 4)

    def _get_status(
        self,
        extraction_result: LandmarkExtractionResult,
        detection_rate: float,
    ) -> tuple[str, str]:
        """Decide whether the extracted sample is usable."""

        if extraction_result.total_frames == 0:
            return "rejected_no_frames", "no_frames_found"

        if extraction_result.detected_frames == 0:
            return "rejected_no_landmarks", "no_hand_landmarks_detected"

        if detection_rate < self.min_detection_rate:
            return "rejected_low_detection_rate", "too_few_detected_frames"

        return "ok", ""

    def _print_summary(self, records: list[MetadataRecord]) -> None:
        """Print a simple extraction summary."""

        status_counts: dict[str, int] = {}

        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1

        print("\nLandmark extraction summary")
        print(f"Total samples: {len(records)}")

        for status, count in status_counts.items():
            print(f"{status}: {count}")
