"""Tests for MetadataWriter.

MetadataWriter used to hand-build CSV rows by joining fields with commas,
so any field containing a comma (error messages from ReaderFactory do)
silently truncated the row on read, with no exception anywhere. This test
locks in that a comma in a field survives a full write/read round trip.

Author: Hristo Hristov
"""

from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
    MetadataWriter,
)
from gesture_transformer.datasets.tensor_extraction.metadata_reader import (
    MetadataReader,
)


def _record(error: str) -> MetadataRecord:
    return MetadataRecord(
        sample_id="s1",
        source_type="video",
        source_name="unit_test",
        label="click",
        raw_label="click",
        path="x.mp4",
        total_frames=5,
        detected_frames=5,
        detection_rate=1.0,
        status="error",
        landmark_path="",
        error=error,
    )


def test_field_containing_a_comma_survives_write_and_read(tmp_path):
    error_with_comma = "Unknown source_type: bad. Available source types: jester, video"
    metadata_path = tmp_path / "metadata.csv"

    MetadataWriter(metadata_path).write([_record(error_with_comma)])
    loaded = MetadataReader(metadata_path)._load_metadata(status="error")

    assert loaded[0].error == error_with_comma
