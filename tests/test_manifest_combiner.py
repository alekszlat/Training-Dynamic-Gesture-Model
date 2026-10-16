"""Tests for ManifestCombiner.

This is the last gate before a manifest reaches disk. If it writes a manifest
that mixes valid rows with silently-dropped invalid ones, every downstream
stage trusts corrupt data with no way to tell. These tests guard the two
behaviors that actually matter: a batch with any invalid row is never
partially written, and a fully valid batch merges both sources correctly.

Author: Hristo Hristov
"""

import csv
from dataclasses import asdict
from pathlib import Path

from gesture_transformer.datasets.manifest.builders.manifest_builder import (
    ManifestAttributes,
)
from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner

RECORDED_LABEL = "click"
JESTER_LABEL = "swiping_left"
SUPPORTED_LABELS = {RECORDED_LABEL, JESTER_LABEL}


def make_row(
    tmp_path: Path,
    sample_id: str,
    label: str = RECORDED_LABEL,
    source_type: str = "video",
    file_exists: bool = True,
) -> ManifestAttributes:
    """Build one manifest row, overriding only what a test cares about."""
    path = tmp_path / (f"{sample_id}.mp4" if source_type == "video" else sample_id)

    if file_exists:
        if source_type == "video":
            path.touch()
        else:
            path.mkdir()

    return ManifestAttributes(
        sample_id=sample_id,
        source_type=source_type,
        source_name="recorded" if source_type == "video" else "jester",
        external_id="" if source_type == "video" else sample_id,
        label=label,
        raw_label=label,
        path=str(path),
    )


def read_manifest_rows(output_path: Path) -> list[dict[str, str]]:
    """Read the manifest CSV the combiner wrote back into rows."""
    with output_path.open(newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def test_build_manifest_rejects_the_whole_batch_when_any_row_is_invalid(tmp_path):
    # arrange
    good = make_row(tmp_path, "good")
    missing_file = make_row(
        tmp_path, "missing", label=JESTER_LABEL, source_type="jester", file_exists=False
    )
    output_path = tmp_path / "manifest.csv"
    combiner = ManifestCombiner(
        datasets_list=[[good], [missing_file]],
        output_path=output_path,
        supported_labels=SUPPORTED_LABELS,
    )

    # act
    built = combiner.build_manifest()

    # assert
    assert built is False
    assert not output_path.exists()


def test_build_manifest_merges_valid_rows_from_both_sources(tmp_path):
    # arrange
    recorded = make_row(tmp_path, "rec_1", label=RECORDED_LABEL)
    jester = make_row(tmp_path, "jester_1", label=JESTER_LABEL, source_type="jester")
    output_path = tmp_path / "manifest.csv"
    combiner = ManifestCombiner(
        datasets_list=[[recorded], [jester]],
        output_path=output_path,
        supported_labels=SUPPORTED_LABELS,
    )

    # act
    built = combiner.build_manifest()

    # assert
    assert built is True
    assert read_manifest_rows(output_path) == [asdict(recorded), asdict(jester)]
