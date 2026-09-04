"""Tests for ManifestCombiner.

This is the last gate before a manifest reaches disk. If it writes a manifest
that mixes valid rows with silently-dropped invalid ones, every downstream
stage trusts corrupt data with no way to tell. These tests guard the two
behaviors that actually matter: a batch with any invalid row is never
partially written, and a fully valid batch merges both sources correctly.

Author: Hristo Hristov
"""

import csv
from pathlib import Path

from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner


def _row(
    tmp_path: Path, sample_id: str, label: str = "click", exists: bool = True
) -> dict:
    path = tmp_path / f"{sample_id}.mp4"
    if exists:
        path.touch()

    return {
        "sample_id": sample_id,
        "source_type": "video",
        "source_name": "unit_test",
        "external_id": sample_id,
        "label": label,
        "raw_label": label,
        "path": str(path),
    }


def test_build_manifest_rejects_the_whole_batch_when_any_row_is_invalid(tmp_path):
    """A single bad row (e.g. a missing file) must fail the whole build, not just be dropped."""
    good = _row(tmp_path, "good")
    missing_file = _row(tmp_path, "missing", exists=False)
    output_path = tmp_path / "manifest.csv"

    combiner = ManifestCombiner(
        [good],
        [missing_file],
        output_path,
        supported_labels={"click"},
    )

    assert combiner.build_manifest() is False
    assert not output_path.exists()


def test_build_manifest_merges_valid_rows_from_both_sources(tmp_path):
    """Recorded and jester rows should end up combined in one CSV when everything validates."""
    recorded = _row(tmp_path, "rec_1", label="click")
    jester = _row(tmp_path, "jester_1", label="swiping_left")
    output_path = tmp_path / "manifest.csv"

    combiner = ManifestCombiner(
        [recorded],
        [jester],
        output_path,
        supported_labels={"click", "swiping_left"},
    )

    assert combiner.build_manifest() is True

    with output_path.open(newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert {row["sample_id"] for row in rows} == {"rec_1", "jester_1"}
