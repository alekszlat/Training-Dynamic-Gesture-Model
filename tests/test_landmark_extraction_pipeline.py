"""Tests for LandmarkExtractionPipeline.

Verifies the pipeline processes samples end to end, writes correct metadata
for both successful and failing samples, and always releases the landmark
extractor, using fakes in place of the manifest, reader and mediapipe
dependencies.

Author: Hristo Hristov
"""

import csv
from pathlib import Path

import numpy as np

from gesture_transformer.datasets.landmark_extraction.landmark_extraction_pipeline import (
    LandmarkExtractionPipeline,
)
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractionResult,
)
from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataWriter,
)
from gesture_transformer.datasets.manifest.sample_manifest_reader import SampleRecord


class FakeManifestReader:
    """Stands in for SampleManifestReader by returning a fixed list of samples."""

    def __init__(self, samples):
        self._samples = samples

    def read_manifest(self):
        """Return the fixed list of samples."""
        return self._samples


class FakeFrameReader:
    """Stands in for a frame reader, either yielding fixed frames or raising an error."""

    def __init__(self, frames=None, error=None):
        self._frames = frames
        self._error = error

    def read_frames(self, path):
        """Return an iterator over the fixed frames, or raise the configured error."""
        if self._error is not None:
            raise self._error
        return iter(self._frames)


class FakeReaderFactory:
    """Stands in for ReaderFactory by mapping source types to fake frame readers."""

    def __init__(self, readers_by_source_type):
        self._readers_by_source_type = readers_by_source_type

    def get_reader(self, source_type):
        """Return the fake frame reader registered for the given source type."""
        return self._readers_by_source_type[source_type]


class FakeLandmarkExtractor:
    """Stands in for LandmarkExtractor so tests don't depend on the real mediapipe model."""

    def __init__(self):
        self.closed = False

    def extract(self, frames):
        """Return a LandmarkExtractionResult sized to the number of frames consumed."""
        frame_list = list(frames)
        total = len(frame_list)

        return LandmarkExtractionResult(
            landmarks=np.zeros((total, 21, 3), dtype=np.float32),
            total_frames=total,
            detected_frames=total,
        )

    def close(self):
        """Mark the extractor as closed."""
        self.closed = True


def test_pipeline_writes_metadata_and_survives_per_sample_failures(tmp_path):
    """Run the pipeline over a successful and a failing sample and verify the metadata output."""
    good_sample = SampleRecord(
        sample_id="sample_good",
        source_type="good_source",
        source_name="unit_test",
        external_id="1",
        label="0",
        raw_label="swipe_left",
        path=Path("good.mp4"),
    )
    bad_sample = SampleRecord(
        sample_id="sample_bad",
        source_type="bad_source",
        source_name="unit_test",
        external_id="2",
        label="1",
        raw_label="swipe_right",
        path=Path("bad.mp4"),
    )

    manifest_reader = FakeManifestReader([good_sample, bad_sample])
    reader_factory = FakeReaderFactory(
        {
            "good_source": FakeFrameReader(
                frames=[np.zeros((4, 4, 3), dtype=np.uint8)] * 5
            ),
            "bad_source": FakeFrameReader(error=RuntimeError("cannot open file")),
        }
    )
    landmark_extractor = FakeLandmarkExtractor()
    landmark_saver = LandmarkSaver(tmp_path / "landmarks")
    metadata_path = tmp_path / "metadata.csv"
    metadata_writer = MetadataWriter(metadata_path)

    pipeline = LandmarkExtractionPipeline(
        manifest_reader=manifest_reader,
        reader_factory=reader_factory,
        landmark_extractor=landmark_extractor,
        landmark_saver=landmark_saver,
        metadata_writer=metadata_writer,
    )

    pipeline.run()

    assert landmark_extractor.closed is True

    with metadata_path.open(newline="") as f:
        written_records = list(csv.DictReader(f))

    assert len(written_records) == 2

    good_record = next(r for r in written_records if r["sample_id"] == "sample_good")
    bad_record = next(r for r in written_records if r["sample_id"] == "sample_bad")

    assert good_record["status"] == "ok"
    assert good_record["detected_frames"] == "5"
    assert Path(good_record["landmark_path"]).exists()

    assert bad_record["status"] == "error"
    assert bad_record["landmark_path"] == ""
    assert "cannot open file" in bad_record["error"]
