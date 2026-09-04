"""Pipeline-wide label consistency tests.

The project's canonical supported labels are defined centrally in config.py.

These tests verify that the pipeline components respect that configuration:

- RecordedManifestBuilder only produces configured labels.
- ManifestCombiner keeps labels outside the configured set out of the output.
- Jester labels are normalized into the project's configured label format.
- LabelEncoder can encode every configured label into a contiguous class index.
- The configured labels remain unique and valid.

The tests intentionally do not inspect the numbered root scripts. Those scripts
act as the composition layer and pass config.SUPPORTED_LABELS into the component
that validates labels before the manifest reaches disk.

Author: Hristo Hristov
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SUPPORTED_LABELS
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner
from gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)
from gesture_transformer.datasets.tensor_extraction.label_encoder import LabelEncoder


def test_supported_labels_are_unique():
    """Configured gesture classes must not contain duplicate labels."""

    assert len(SUPPORTED_LABELS) == len(set(SUPPORTED_LABELS))


def test_supported_labels_use_internal_label_format():
    """Configured labels must already use the normalized internal format."""

    mapper = LabelMapper()

    for label in SUPPORTED_LABELS:
        assert mapper.converter_label(label) == label, (
            f"Configured label '{label}' is not in normalized internal format."
        )


def test_recorded_builder_produces_supported_labels(tmp_path):
    """Recorded folders named after configured classes must produce those labels."""

    samples_dir = tmp_path / "recorded"

    for label in SUPPORTED_LABELS:
        folder = samples_dir / label
        folder.mkdir(parents=True)
        (folder / "001.mp4").touch()

    builder = RecordedManifestBuilder(samples_dir=samples_dir)

    samples = builder.build()

    produced_labels = {sample["label"] for sample in samples}

    assert produced_labels == set(SUPPORTED_LABELS), (
        "RecordedManifestBuilder did not produce exactly the configured labels. "
        f"Expected: {set(SUPPORTED_LABELS)}, "
        f"got: {produced_labels}"
    )


def test_combiner_filters_unsupported_recorded_label(tmp_path):
    """Unsupported recorded labels must not reach the manifest on disk."""

    samples_dir = tmp_path / "recorded"

    supported_folder = samples_dir / next(iter(SUPPORTED_LABELS))
    supported_folder.mkdir(parents=True)
    (supported_folder / "001.mp4").touch()

    unsupported_folder = samples_dir / "unsupported_gesture"
    unsupported_folder.mkdir()
    (unsupported_folder / "001.mp4").touch()

    recorded_samples = RecordedManifestBuilder(samples_dir=samples_dir).build()
    output_path = tmp_path / "manifest.csv"
    combiner = ManifestCombiner(
        recorded_list=recorded_samples,
        jester_list=[],
        output_path=output_path,
        supported_labels=SUPPORTED_LABELS,
    )

    assert combiner.build_manifest() is True

    manifest_text = output_path.read_text(encoding="utf-8")

    assert unsupported_folder.name not in manifest_text
    assert supported_folder.name in manifest_text


def test_jester_style_labels_normalize_to_internal_format():
    """Human-readable Jester labels must normalize to project label format."""

    mapper = LabelMapper()

    jester_labels = {
        "Swiping Left",
        "Swiping Right",
        "Swiping Up",
        "Swiping Down",
    }

    mapped_labels = {mapper.converter_label(label) for label in jester_labels}

    assert mapped_labels <= set(SUPPORTED_LABELS), (
        "Jester labels mapped to values outside config.SUPPORTED_LABELS: "
        f"{mapped_labels - set(SUPPORTED_LABELS)}"
    )


def test_label_encoder_supports_all_configured_labels():
    """Every configured gesture must be encodable by the tensor-building stage."""

    encoder = LabelEncoder(list(SUPPORTED_LABELS))

    encoded_labels = {label: encoder.encode(label) for label in SUPPORTED_LABELS}

    assert set(encoded_labels) == set(SUPPORTED_LABELS)


def test_label_encoder_produces_contiguous_class_indices():
    """Class indices must be contiguous from 0 to number_of_classes - 1."""

    encoder = LabelEncoder(list(SUPPORTED_LABELS))

    indices = {encoder.encode(label) for label in SUPPORTED_LABELS}

    expected_indices = set(range(len(SUPPORTED_LABELS)))

    assert indices == expected_indices, (
        "LabelEncoder produced non-contiguous class indices. "
        f"Expected {expected_indices}, got {indices}"
    )


def test_label_mapping_contains_exactly_supported_labels():
    """The generated label mapping must contain exactly the configured classes."""

    encoder = LabelEncoder(list(SUPPORTED_LABELS))

    mapping = encoder.mapping()

    assert set(mapping.keys()) == set(SUPPORTED_LABELS)
    assert len(mapping) == len(SUPPORTED_LABELS)
