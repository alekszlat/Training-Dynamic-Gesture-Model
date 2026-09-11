"""Tests for LandmarkExtractionPipeline.

Verifies the pipeline processes samples end to end, writes correct metadata
for both successful and failing samples, and always releases the landmark
extractor, using fakes in place of the manifest, reader and mediapipe
dependencies.

The fakes are checked against the real classes they stand in for (U23). A fake
whose interface drifts from the real one keeps every test in this file green
while production breaks, and nothing else in the suite would notice.

Author: Hristo Hristov
"""

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest

from gesture_transformer.datasets.landmark_extraction.landmark_extraction_pipeline import (
    LandmarkExtractionPipeline,
)
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractionResult,
    LandmarkExtractor,
)
from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataWriter,
)
from gesture_transformer.datasets.manifest.sample_manifest_reader import (
    SampleManifestReader,
    SampleRecord,
)
from gesture_transformer.datasets.readers.base_reader import (
    FrameReader,
)
from gesture_transformer.datasets.readers.reader_factory import (
    ReaderFactory,
)

METADATA_FILENAME = "metadata.csv"
LANDMARKS_DIRNAME = "landmarks"
GOOD_SOURCE_TYPE = "good_source"
BAD_SOURCE_TYPE = "bad_source"
FRAME_COUNT = 5
READER_FAILURE_MESSAGE = "cannot open file"


class FakeManifestReader(SampleManifestReader):
    """Stands in for SampleManifestReader by returning a fixed list of samples."""

    def __init__(self, samples):
        self._samples = samples

    def read_manifest(self):
        """Return the fixed list of samples."""
        return self._samples


class FakeFrameReader:
    """Stands in for a frame reader, either yielding fixed frames or raising an error."""

    def __init__(self, frames=None, error=None):
        self._frames = frames if frames is not None else []
        self._error = error

    def read_frames(self, path):
        """Return an iterator over the fixed frames, or raise the configured error."""
        if self._error is not None:
            raise self._error
        return iter(self._frames)


class FakeReaderFactory(ReaderFactory):
    """Stands in for ReaderFactory by mapping source types to fake frame readers."""

    def __init__(self, readers_by_source_type):
        self._readers_by_source_type = readers_by_source_type

    def get_reader(self, source_type):
        """Return the fake frame reader registered for the given source type."""
        return self._readers_by_source_type[source_type]


class FakeLandmarkExtractor(LandmarkExtractor):
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


def make_sample(**overrides) -> SampleRecord:
    """Build a SampleRecord, overriding only the fields a test cares about."""
    fields = {
        "sample_id": "sample_good",
        "source_type": GOOD_SOURCE_TYPE,
        "source_name": "unit_test",
        "external_id": "1",
        "label": "0",
        "raw_label": "swipe_left",
        "path": Path("good.mp4"),
    }

    return SampleRecord(**{**fields, **overrides})


def make_pipeline(
    tmp_path: Path,
    samples: list[SampleRecord],
    landmark_extractor: FakeLandmarkExtractor,
) -> LandmarkExtractionPipeline:
    """Build a pipeline wired to fakes, writing its output under tmp_path."""
    readers_by_source_type = {
        GOOD_SOURCE_TYPE: FakeFrameReader(
            frames=[np.zeros((4, 4, 3), dtype=np.uint8)] * FRAME_COUNT
        ),
        BAD_SOURCE_TYPE: FakeFrameReader(error=RuntimeError(READER_FAILURE_MESSAGE)),
    }

    return LandmarkExtractionPipeline(
        manifest_reader=FakeManifestReader(samples),
        reader_factory=FakeReaderFactory(readers_by_source_type),
        landmark_extractor=landmark_extractor,
        landmark_saver=LandmarkSaver(tmp_path / LANDMARKS_DIRNAME),
        metadata_writer=MetadataWriter(tmp_path / METADATA_FILENAME),
    )


def read_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    """Read the metadata CSV the pipeline wrote back into rows."""
    with metadata_path.open(newline="") as metadata_file:
        return list(csv.DictReader(metadata_file))


def test_pipeline_writes_an_ok_metadata_row_for_a_successful_sample(tmp_path: Path):
    # arrange
    sample = make_sample()
    pipeline = make_pipeline(tmp_path, [sample], FakeLandmarkExtractor())

    # act
    pipeline.run()

    # assert
    expected_row = {
        "sample_id": "sample_good",
        "source_type": GOOD_SOURCE_TYPE,
        "source_name": "unit_test",
        "label": "0",
        "raw_label": "swipe_left",
        "path": "good.mp4",
        "total_frames": str(FRAME_COUNT),
        "detected_frames": str(FRAME_COUNT),
        "detection_rate": "1.00",
        "status": "ok",
        "landmark_path": str(tmp_path / LANDMARKS_DIRNAME / "sample_good.npy"),
        "error": "",
    }

    assert read_metadata_rows(tmp_path / METADATA_FILENAME) == [expected_row]


def test_pipeline_saves_landmarks_for_a_successful_sample(tmp_path: Path):
    # arrange
    sample = make_sample()
    pipeline = make_pipeline(tmp_path, [sample], FakeLandmarkExtractor())

    # act
    pipeline.run()

    # assert
    saved_landmarks = np.load(tmp_path / LANDMARKS_DIRNAME / "sample_good.npy")

    assert saved_landmarks.shape == (FRAME_COUNT, 21, 3)


def test_pipeline_writes_an_error_metadata_row_when_the_reader_fails(tmp_path: Path):
    # arrange
    sample = make_sample(
        sample_id="sample_bad",
        source_type=BAD_SOURCE_TYPE,
        external_id="2",
        label="1",
        raw_label="swipe_right",
        path=Path("bad.mp4"),
    )
    pipeline = make_pipeline(tmp_path, [sample], FakeLandmarkExtractor())

    # act
    pipeline.run()

    # assert
    expected_row = {
        "sample_id": "sample_bad",
        "source_type": BAD_SOURCE_TYPE,
        "source_name": "unit_test",
        "label": "1",
        "raw_label": "swipe_right",
        "path": "bad.mp4",
        "total_frames": "0",
        "detected_frames": "0",
        "detection_rate": "0.00",
        "status": "error",
        "landmark_path": "",
        "error": READER_FAILURE_MESSAGE,
    }

    assert read_metadata_rows(tmp_path / METADATA_FILENAME) == [expected_row]


def test_pipeline_keeps_processing_after_a_sample_fails(tmp_path: Path):
    # arrange
    good_sample = make_sample()
    bad_sample = make_sample(
        sample_id="sample_bad",
        source_type=BAD_SOURCE_TYPE,
        path=Path("bad.mp4"),
    )
    pipeline = make_pipeline(
        tmp_path, [bad_sample, good_sample], FakeLandmarkExtractor()
    )

    # act
    pipeline.run()

    # assert
    written_rows = read_metadata_rows(tmp_path / METADATA_FILENAME)
    written_statuses = [(row["sample_id"], row["status"]) for row in written_rows]

    assert written_statuses == [("sample_bad", "error"), ("sample_good", "ok")]


def test_pipeline_closes_the_landmark_extractor_after_a_sample_fails(tmp_path: Path):
    # arrange
    sample = make_sample(source_type=BAD_SOURCE_TYPE, path=Path("bad.mp4"))
    landmark_extractor = FakeLandmarkExtractor()
    pipeline = make_pipeline(tmp_path, [sample], landmark_extractor)

    # act
    pipeline.run()

    # assert
    assert landmark_extractor.closed is True


# U23: the fakes above stand in for these classes. Instantiating the real
# LandmarkExtractor loads a 7.5M mediapipe model, so the contract is checked at
# the interface rather than by running one suite against both implementations.
FAKE_REAL_PAIRS = [
    (FakeManifestReader, SampleManifestReader),
    (FakeFrameReader, FrameReader),
    (FakeReaderFactory, ReaderFactory),
    (FakeLandmarkExtractor, LandmarkExtractor),
]


def _public_methods(cls) -> dict:
    """Public methods of a class, keyed by name."""

    return {
        name: function
        for name, function in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def _parameters(function) -> list:
    """Parameter names and kinds, which is the part a caller depends on."""

    return [
        (parameter.name, parameter.kind)
        for parameter in inspect.signature(function).parameters.values()
    ]


@pytest.mark.parametrize(
    "fake, real",
    FAKE_REAL_PAIRS,
    ids=[fake.__name__ for fake, _ in FAKE_REAL_PAIRS],
)
def test_fake_matches_the_real_public_interface(fake, real):
    # arrange
    expected = {
        name: _parameters(function) for name, function in _public_methods(real).items()
    }

    # act
    fake_methods = _public_methods(fake)

    # assert
    actual = {
        name: _parameters(fake_methods[name]) if name in fake_methods else None
        for name in expected
    }

    assert actual == expected
