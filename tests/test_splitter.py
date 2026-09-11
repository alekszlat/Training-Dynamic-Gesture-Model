"""Tests for Splitter.

A train/validation split that leaks a sample into both sides doesn't crash
and doesn't look wrong in a print statement - it just quietly inflates
validation accuracy. These tests exist to catch exactly that class of bug,
plus the stratification guarantee the split is supposed to provide.

Author: Hristo Hristov
"""

from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from gesture_transformer.datasets.tensor_extraction.splitter import Splitter

CLICK_LABEL = "click"
SWIPE_LABEL = "swiping_left"
CLICK_SAMPLE_COUNT = 10
SWIPE_SAMPLE_COUNT = 5
VAL_RATIO = 0.2
RANDOM_SEED = 42


def make_record(sample_id: str, label: str) -> MetadataRecord:
    """Build a usable metadata record, varying only id and label."""
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


def make_records() -> list[MetadataRecord]:
    """Build an imbalanced two-label set of records to split."""
    click_records = [
        make_record(f"{CLICK_LABEL}_{index}", CLICK_LABEL)
        for index in range(CLICK_SAMPLE_COUNT)
    ]
    swipe_records = [
        make_record(f"{SWIPE_LABEL}_{index}", SWIPE_LABEL)
        for index in range(SWIPE_SAMPLE_COUNT)
    ]

    return click_records + swipe_records


def test_split_puts_no_sample_in_both_train_and_validation():
    # arrange
    records = make_records()

    # act
    result = Splitter(val_ratio=VAL_RATIO, random_seed=RANDOM_SEED).split(records)

    # assert
    # If Splitter grows a third split (e.g. train/dev/test), add a set for it
    # and assert isdisjoint() between every pair, not just train vs val.
    train_ids = {record.sample_id for record in result.train_records}
    val_ids = {record.sample_id for record in result.val_records}

    assert train_ids.isdisjoint(val_ids)


def test_split_keeps_every_sample_in_exactly_one_side():
    # arrange
    records = make_records()

    # act
    result = Splitter(val_ratio=VAL_RATIO, random_seed=RANDOM_SEED).split(records)

    # assert
    split_ids = sorted(
        record.sample_id for record in result.train_records + result.val_records
    )

    assert split_ids == sorted(record.sample_id for record in records)


def test_split_is_stratified_so_every_label_reaches_validation():
    # arrange
    records = make_records()

    # act
    result = Splitter(val_ratio=VAL_RATIO, random_seed=RANDOM_SEED).split(records)

    # assert
    val_labels = {record.label for record in result.val_records}

    assert val_labels == {CLICK_LABEL, SWIPE_LABEL}
