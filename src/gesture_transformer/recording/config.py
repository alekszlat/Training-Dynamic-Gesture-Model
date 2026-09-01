"""
Recording session settings: where takes are written and what counts as usable.

Dependencies:
    gesture_transformer.datasets.manifest: source of the supported label names.

Author:
    Hristo Hristov
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from gesture_transformer.datasets.manifest.manifest_combiner import SUPPORTED_LABELS

# Sorted so member order matches the class indices LabelEncoder assigns.
Label = Enum(
    value="Label",
    names={
        label.upper(): label  # key: value
        for label in sorted(SUPPORTED_LABELS)
    },
)


class Split(Enum):
    """Dataset split a recorded take belongs to."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


@dataclass
class Config:
    """
    Settings for one recording session.

    Attributes:
        output_dir: Root folder takes are written under.
        active_split: Split the current takes belong to.
        manifest_path: Manifest the session's takes are appended to.
        metadata_path: Landmark metadata the session's takes are appended to.
        landmarks_dir: Folder the extracted landmark arrays are written to.
        active_label: Gesture being recorded. Must be set before recording.
        min_frames: Fewest frames a take may have and still be usable.
        min_detection_rate: Lowest fraction of frames with a detected hand, 0 to 1.
        model_path: MediaPipe hand landmarker model used to check takes.
        webcam_id: Index of the camera to open.
        default_fps: Frame rate used when the camera reports an implausible one.
        max_fps: Highest frame rate accepted from the camera.
        max_consecutive_read_failures: Failed reads in a row before a take ends.
    """

    output_dir: Path = Path("data/raw/recorded")
    active_split: Split = Split.TRAIN
    manifest_path: Path = Path("data/manifests/recorded_samples.csv")
    metadata_path: Path = Path("data/processed/recorded_metadata.csv")
    landmarks_dir: Path = Path("data/interim/landmarks/recorded")
    active_label: Label | None = None
    min_frames: int = 40
    min_detection_rate: float = 0.5
    model_path: Path = Path("src/gesture_transformer/models/hand_landmarker.task")
    webcam_id: int = 0
    default_fps: float = 30.0
    max_fps: float = 120.0
    max_consecutive_read_failures: int = 30
