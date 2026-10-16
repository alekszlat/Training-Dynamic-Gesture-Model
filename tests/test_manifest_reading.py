"""Tests for SampleManifestReader and ReaderFactory.

These two classes sit right before a sample's raw frames get touched: one
turns a manifest CSV back into typed records, the other picks which reader
reads them. A column silently read into the wrong field, or an unknown
source type failing without saying which one, would surface far away from
its actual cause. These tests pin down both contracts directly.

Author: Hristo Hristov
"""

from pathlib import Path

import pytest

from gesture_transformer.datasets.manifest.sample_manifest_reader import (
    SampleManifestReader,
    SampleRecord,
)
from gesture_transformer.datasets.readers.reader_factory import ReaderFactory

MANIFEST_HEADER = "sample_id,source_type,source_name,external_id,label,raw_label,path\n"
SAMPLE_PATH = "data/raw/recorded/click/001.mp4"
UNKNOWN_SOURCE_TYPE = "unknown_source"


def test_sample_manifest_reader_parses_csv_rows_into_typed_records(tmp_path):
    # arrange
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        MANIFEST_HEADER + f"sample_1,video,unit_test,1,click,click,{SAMPLE_PATH}\n",
        encoding="utf-8",
    )

    # act
    records = SampleManifestReader(manifest_path).read_manifest()

    # assert
    expected_record = SampleRecord(
        sample_id="sample_1",
        source_type="video",
        source_name="unit_test",
        external_id="1",
        label="click",
        raw_label="click",
        path=Path(SAMPLE_PATH),
    )

    assert records == [expected_record]


def test_reader_factory_raises_clear_error_for_unknown_source_type():
    # arrange
    factory = ReaderFactory()

    # act / assert
    with pytest.raises(ValueError, match=UNKNOWN_SOURCE_TYPE):
        factory.get_reader(UNKNOWN_SOURCE_TYPE)
