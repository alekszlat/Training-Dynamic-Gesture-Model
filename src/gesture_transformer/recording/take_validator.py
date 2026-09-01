"""
Checks whether a recorded take has enough frames and enough detected hands to keep.

Dependencies:
    gesture_transformer.datasets: frame reading and MediaPipe landmark extraction.

Author:
    Hristo Hristov
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractor,
)
from gesture_transformer.datasets.readers.video_reader import VideoReader
from gesture_transformer.recording.config import Config


@dataclass(frozen=True)
class TakeValidationResult:
    """Outcome of checking one recorded take."""

    landmarks: np.ndarray
    total_frames: int
    detected_frames: int
    detection_rate: float
    is_valid: bool
    reason: str


class TakeValidator:
    """Decides whether a recorded take is usable, using the thresholds in Config."""

    def __init__(
        self,
        config: Config,
        landmark_extractor: LandmarkExtractor,
        video_reader: VideoReader,
    ):
        """
        Store the settings and the collaborators used to read a take.

        Args:
            config: Recording session settings.
            landmark_extractor: Extractor run over the take. The caller closes it.
            video_reader: Reader for the recorded mp4.
        """

        self.config = config
        self.landmark_extractor = landmark_extractor
        self.video_reader = video_reader

    def validate(self, take_path: Path) -> TakeValidationResult:
        """
        Check one take against the frame count and detection rate thresholds.

        Args:
            take_path: Path to the recorded mp4.

        Returns:
            The landmarks, frame counts, detection rate and the keep or reject
            decision. The landmarks are kept so a usable take does not have to be
            read a second time to save them.
        """

        frames = self.video_reader.read_frames(take_path)
        extraction_result = self.landmark_extractor.extract(frames)

        detection_rate = self._calculate_detection_rate(
            total_frames=extraction_result.total_frames,
            detected_frames=extraction_result.detected_frames,
        )

        is_valid, reason = self._get_decision(
            total_frames=extraction_result.total_frames,
            detection_rate=detection_rate,
        )

        return TakeValidationResult(
            landmarks=extraction_result.landmarks,
            total_frames=extraction_result.total_frames,
            detected_frames=extraction_result.detected_frames,
            detection_rate=detection_rate,
            is_valid=is_valid,
            reason=reason,
        )

    def _calculate_detection_rate(
        self,
        total_frames: int,
        detected_frames: int,
    ) -> float:
        """
        Calculate detected_frames / total_frames safely.

        Args:
            total_frames: Frames read from the take.
            detected_frames: Frames a hand was found in.

        Returns:
            Share of frames with a detected hand, 0 to 1.
        """

        if total_frames == 0:
            return 0.0

        return round(detected_frames / total_frames, 4)

    def _get_decision(
        self,
        total_frames: int,
        detection_rate: float,
    ) -> tuple[bool, str]:
        """
        Apply the keep or reject rules to one take.

        Args:
            total_frames: Frames read from the take.
            detection_rate: Share of frames with a detected hand, 0 to 1.

        Returns:
            Whether to keep the take, and why it was rejected if it was not.
        """

        if total_frames < self.config.min_frames:
            return False, f"only {total_frames} frames, need {self.config.min_frames}"

        if detection_rate < self.config.min_detection_rate:
            return (
                False,
                f"detection rate {detection_rate}, need {self.config.min_detection_rate}",
            )

        return True, ""
