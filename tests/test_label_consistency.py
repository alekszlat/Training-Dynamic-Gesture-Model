"""Pipeline-wide label consistency tests.

Each pipeline stage has its own idea of which labels are valid:
  - RecordedManifestBuilder / JesterManifestBuilder produce labels (stage 1)
  - ManifestCombiner.SUPPORTED_LABELS accepts or rejects them (stage 1 -> 2)
  - 03_build_tensors.py's LABEL_LIST encodes them into tensors (stage 3)

SUPPORTED_LABELS and LABEL_LIST used to be independent, hand-typed sets that
disagreed on "doing_other_things". These tests assert every stage boundary
agrees, so a label accepted by one stage can never be rejected by a later
one.

Author: Hristo Hristov
"""

import importlib.util
from pathlib import Path

from gesture_transformer.datasets.manifest.jester_manifest_builder import (
    SUPPORTED_JESTER_LABELS,
)
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from gesture_transformer.datasets.manifest.manifest_combiner import SUPPORTED_LABELS
from gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTED_GESTURE_FOLDERS = [
    "swiping_up",
    "swiping_down",
    "swiping_left",
    "swiping_right",
    "click",
]


def _load_label_list_from_build_tensors_script() -> list[str]:
    """Import LABEL_LIST from 03_build_tensors.py without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(
        "build_tensors_script", REPO_ROOT / "03_build_tensors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LABEL_LIST


def test_documented_folder_layout_produces_supported_labels(tmp_path):
    """Folders named exactly as documented in the builder's docstring should yield valid labels.

    ManifestCombiner's SUPPORTED_LABELS assets folder names as "swiping_up",
    "swiping_down", while documetation and every other file lists them as
    "swipe_up", "swipe_down", etc. This test ensures unified label names are used
    across the pipeline.
    """
    samples_dir = tmp_path / "recorded"
    for folder_name in DOCUMENTED_GESTURE_FOLDERS:
        folder = samples_dir / folder_name
        folder.mkdir(parents=True)
        (folder / "001.mp4").touch()

    builder = RecordedManifestBuilder(samples_dir)
    samples = builder.build()

    unsupported = {sample["label"] for sample in samples} - SUPPORTED_LABELS
    assert not unsupported, (
        f"Documented folder layout produced labels not in SUPPORTED_LABELS: {unsupported}"
    )


def test_jester_labels_map_into_supported_labels():
    """Every label JesterManifestBuilder can emit must be accepted by ManifestCombiner."""
    mapper = LabelMapper()
    mapped_labels = {mapper.converter_label(raw) for raw in SUPPORTED_JESTER_LABELS}

    unsupported = mapped_labels - SUPPORTED_LABELS
    assert not unsupported, (
        f"Jester labels map to values ManifestCombiner rejects: {unsupported}"
    )


def test_tensor_builder_label_list_matches_supported_labels():
    """Every label ManifestCombiner accepts must be usable by the tensor-building stage."""
    label_list = _load_label_list_from_build_tensors_script()

    assert set(label_list) == SUPPORTED_LABELS, (
        "03_build_tensors.py's LABEL_LIST has drifted from "
        f"ManifestCombiner.SUPPORTED_LABELS: {set(label_list) ^ SUPPORTED_LABELS}"
    )
