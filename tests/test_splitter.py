"""Tests for Splitter.

A train/validation split that leaks a sample into both sides doesn't crash
and doesn't look wrong in a print statement - it just quietly inflates
validation accuracy. This test exists to catch exactly that class of bug,
plus the stratification guarantee the split is supposed to provide.

Author: Hristo Hristov
"""

from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from gesture_transformer.datasets.tensor_extraction.splitter import Splitter


def _record(sample_id: str, label: str) -> MetadataRecord:
    return MetadataRecord(
        sample_id=sample_id,
        source_type="video",
        source_name="unit_test",
        label=label,
        raw_label=label,
        path=f"{sample_id}.mp4",
        total_frames=10,
        detected_frames=10,
        detection_rate=1.0,
        status="ok",
        landmark_path=f"{sample_id}.npy",
        error="",
    )


def test_split_has_no_leakage_and_keeps_every_label_in_both_sets():
    records = [_record(f"click_{i}", "click") for i in range(10)]
    records += [_record(f"swipe_{i}", "swiping_left") for i in range(5)]

    result = Splitter(val_ratio=0.2, random_seed=42).split(records)

    train_ids = {r.sample_id for r in result.train_records}
    val_ids = {r.sample_id for r in result.val_records}

    # If Splitter grows a third split (e.g. train/dev/test), add a set for it
    # and assert isdisjoint() between every pair, not just train vs val.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {r.sample_id for r in records}

    val_labels = {r.label for r in result.val_records}
    assert val_labels == {"click", "swiping_left"}
