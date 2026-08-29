"""Pipeline-wide label consistency tests.

The project's canonical supported labels are defined centrally in config.py.

These tests verify that every pipeline stage remains consistent with that
central configuration:

  - RecordedManifestBuilder produces labels accepted by config.SUPPORTED_LABELS.
  - Jester raw labels map to labels accepted by config.SUPPORTED_LABELS.
  - 03_build_tensors.py's LABEL_LIST matches config.SUPPORTED_LABELS.

This prevents individual pipeline stages from silently drifting away from
the project's configured gesture classes.

Author: Hristo Hristov
"""

import importlib.util
from pathlib import Path

from config import SUPPORTED_LABELS
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# Folder names expected by RecordedManifestBuilder.
DOCUMENTED_GESTURE_FOLDERS = [
    "swiping_up",
    "swiping_down",
    "swiping_left",
    "swiping_right",
    "click",
]


# Raw Jester labels used by this project.
#
# Jester does not provide a "click" gesture, so only the four swipe
# gestures are expected to map into the project's supported labels.
JESTER_GESTURE_LABELS = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Swiping Down",
]


def _load_label_list_from_build_tensors_script() -> list[str]:
    """Load LABEL_LIST from 03_build_tensors.py without running its main block."""

    spec = importlib.util.spec_from_file_location(
        "build_tensors_script",
        REPO_ROOT / "03_build_tensors.py",
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load 03_build_tensors.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.LABEL_LIST


def test_documented_folder_layout_produces_supported_labels(tmp_path):
    """Recorded gesture folders must produce centrally supported labels."""

    samples_dir = tmp_path / "recorded"

    for folder_name in DOCUMENTED_GESTURE_FOLDERS:
        folder = samples_dir / folder_name
        folder.mkdir(parents=True)
        (folder / "001.mp4").touch()

    builder = RecordedManifestBuilder(samples_dir)
    samples = builder.build()

    produced_labels = {sample["label"] for sample in samples}
    unsupported = produced_labels - SUPPORTED_LABELS

    assert not unsupported, (
        "RecordedManifestBuilder produced labels not present in "
        f"config.SUPPORTED_LABELS: {unsupported}"
    )


def test_jester_labels_map_into_supported_labels():
    """Jester gesture labels must map into centrally supported labels."""

    mapper = LabelMapper()

    mapped_labels = {
        mapper.converter_label(raw_label) for raw_label in JESTER_GESTURE_LABELS
    }

    unsupported = mapped_labels - SUPPORTED_LABELS

    assert not unsupported, (
        "Jester labels mapped to values not present in "
        f"config.SUPPORTED_LABELS: {unsupported}"
    )


def test_tensor_builder_label_list_matches_supported_labels():
    """Tensor-building labels must exactly match the central configuration."""

    label_list = _load_label_list_from_build_tensors_script()

    assert set(label_list) == SUPPORTED_LABELS, (
        "03_build_tensors.py's LABEL_LIST has drifted from "
        "config.SUPPORTED_LABELS: "
        f"{set(label_list) ^ SUPPORTED_LABELS}"
    )
