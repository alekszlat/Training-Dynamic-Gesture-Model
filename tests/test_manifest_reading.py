"""Tests for SampleManifestReader and ReaderFactory.

These two classes sit right before a sample's raw frames get touched: one
turns a manifest CSV back into typed records, the other picks which reader
reads them. A column silently read into the wrong field, or an unknown
source type failing without saying which one, would surface far away from
its actual cause. These tests pin down both contracts directly.

Author: Hristo Hristov
"""

import csv
from pathlib import Path

import pytest

from gesture_transformer.datasets.manifest.sample_manifest_reader import (
    SampleManifestReader,
)
from gesture_transformer.datasets.readers.reader_factory import ReaderFactory


def test_sample_manifest_reader_parses_csv_rows_into_typed_records(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "sample_id",
                "source_type",
                "source_name",
                "external_id",
                "label",
                "raw_label",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_1",
                "source_type": "video",
                "source_name": "unit_test",
                "external_id": "1",
                "label": "click",
                "raw_label": "click",
                "path": "data/raw/recorded/click/001.mp4",
            }
        )

    records = SampleManifestReader(manifest_path).read_manifest()

    assert len(records) == 1
    assert records[0].sample_id == "sample_1"
    assert records[0].label == "click"
    assert records[0].path == Path("data/raw/recorded/click/001.mp4")


def test_reader_factory_raises_clear_error_for_unknown_source_type():
    factory = ReaderFactory()

    with pytest.raises(ValueError, match="unknown_source"):
        factory.get_reader("unknown_source")
