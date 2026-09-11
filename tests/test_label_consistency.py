"""Pipeline-wide label consistency tests.

The project's canonical supported labels are defined centrally in config.py.

These tests verify that the pipeline components respect that configuration:

- RecordedManifestBuilder only produces configured labels.
- ManifestCombiner keeps labels outside the configured set out of the output.
- Jester labels are normalized into the project's configured label format.
- LabelEncoder can encode every configured label into a contiguous class index.

The tests intentionally do not inspect the numbered root scripts. Those scripts
act as the composition layer and pass config.SUPPORTED_LABELS into the component
that validates labels before the manifest reaches disk.

Author: Hristo Hristov
"""

import csv

import pytest

from config import SUPPORTED_LABELS
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner
from gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)
from gesture_transformer.datasets.tensor_extraction.label_encoder import LabelEncoder

# SUPPORTED_LABELS is a set, so iteration order varies with PYTHONHASHSEED.
# Everything below works from the sorted form to stay deterministic.
SORTED_SUPPORTED_LABELS = sorted(SUPPORTED_LABELS)
UNSUPPORTED_LABEL = "unsupported_gesture"
VIDEO_FILENAME = "001.mp4"
JESTER_LABELS = ["Swiping Left", "Swiping Right", "Swiping Up", "Swiping Down"]


@pytest.fixture
def recorded_samples_dir(tmp_path):
    """A recorded-samples tree with one folder per configured label."""
    samples_dir = tmp_path / "recorded"

    for label in SORTED_SUPPORTED_LABELS:
        label_folder = samples_dir / label
        label_folder.mkdir(parents=True)
        (label_folder / VIDEO_FILENAME).touch()

    return samples_dir


@pytest.mark.parametrize("label", SORTED_SUPPORTED_LABELS)
def test_supported_label_is_already_in_internal_format(label):
    # arrange
    mapper = LabelMapper()

    # act
    normalized_label = mapper.converter_label(label)

    # assert
    assert normalized_label == label


def test_recorded_builder_produces_exactly_the_supported_labels(recorded_samples_dir):
    # arrange
    builder = RecordedManifestBuilder(samples_dir=recorded_samples_dir)

    # act
    samples = builder.build()

    # assert
    produced_labels = {sample["label"] for sample in samples}

    assert produced_labels == set(SUPPORTED_LABELS)


def test_combiner_keeps_an_unsupported_recorded_label_out_of_the_manifest(tmp_path):
    # arrange
    samples_dir = tmp_path / "recorded"
    supported_label = SORTED_SUPPORTED_LABELS[0]

    supported_folder = samples_dir / supported_label
    supported_folder.mkdir(parents=True)
    (supported_folder / VIDEO_FILENAME).touch()

    unsupported_folder = samples_dir / UNSUPPORTED_LABEL
    unsupported_folder.mkdir(parents=True)
    (unsupported_folder / VIDEO_FILENAME).touch()

    recorded_samples = RecordedManifestBuilder(samples_dir=samples_dir).build()
    output_path = tmp_path / "manifest.csv"
    combiner = ManifestCombiner(
        recorded_list=recorded_samples,
        jester_list=[],
        output_path=output_path,
        supported_labels=SUPPORTED_LABELS,
    )

    # act
    built = combiner.build_manifest()

    # assert
    with output_path.open(newline="", encoding="utf-8") as manifest_file:
        written_labels = {row["label"] for row in csv.DictReader(manifest_file)}

    assert built is True
    assert written_labels == {supported_label}


@pytest.mark.parametrize("jester_label", JESTER_LABELS)
def test_jester_style_label_normalizes_into_a_supported_label(jester_label):
    # arrange
    mapper = LabelMapper()

    # act
    normalized_label = mapper.converter_label(jester_label)

    # assert
    assert normalized_label in SUPPORTED_LABELS


@pytest.mark.parametrize("label", SORTED_SUPPORTED_LABELS)
def test_label_encoder_round_trips_every_configured_label(label):
    # arrange
    encoder = LabelEncoder(SORTED_SUPPORTED_LABELS)

    # act
    encoded_index = encoder.encode(label)

    # assert
    assert encoder.decode(encoded_index) == label


def test_label_encoder_produces_contiguous_class_indices():
    # arrange
    encoder = LabelEncoder(SORTED_SUPPORTED_LABELS)

    # act
    indices = {encoder.encode(label) for label in SORTED_SUPPORTED_LABELS}

    # assert
    assert indices == set(range(len(SUPPORTED_LABELS)))


def test_label_mapping_contains_exactly_the_supported_labels():
    # arrange
    encoder = LabelEncoder(SORTED_SUPPORTED_LABELS)

    # act
    mapping = encoder.mapping()

    # assert
    assert set(mapping) == set(SUPPORTED_LABELS)
