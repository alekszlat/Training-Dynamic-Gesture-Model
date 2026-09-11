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

ERROR_WITH_COMMA = "Unknown source_type: bad. Available source types: jester, video"


def make_record(error: str) -> MetadataRecord:
    """Build a failed-extraction metadata record, varying only the error text."""
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
    # arrange
    record = make_record(ERROR_WITH_COMMA)
    metadata_path = tmp_path / "metadata.csv"

    # act
    MetadataWriter(metadata_path).write([record])
    loaded = MetadataReader(metadata_path)._load_metadata(status="error")

    # assert
    assert loaded == [record]
