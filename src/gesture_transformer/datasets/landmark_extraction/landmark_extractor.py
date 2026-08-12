from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

from gesture_transformer.datasets.readers.base_reader import Frame

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass(frozen=True)
class LandmarkExtractionResult:
    """Result of extracting landmarks from one gesture sample."""

    landmarks: np.ndarray
    total_frames: int
    detected_frames: int


class LandmarkExtractor:
    """Extracts one-hand MediaPipe landmarks from a sequence of frames."""

    def __init__(
        self,
        model_path: Path,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = HandLandmarker.create_from_options(options)

    def extract(self, frames: Iterable[Frame]) -> LandmarkExtractionResult:
        """
        Extract landmarks from one gesture sample.

        Returns:
            landmarks: [total_frames, 21, 3]

        If MediaPipe does not detect a hand in a frame,
        a zero frame with shape [21, 3] is added.
        """

        sample_landmarks: list[np.ndarray] = []
        total_frames = 0
        detected_frames = 0

        for frame in frames:
            total_frames += 1

            frame_landmarks = self._extract_frame_landmarks(frame)

            if frame_landmarks is None:
                frame_landmarks = np.zeros((21, 3), dtype=np.float32)
            else:
                detected_frames += 1

            sample_landmarks.append(frame_landmarks)

        if not sample_landmarks:
            landmarks = np.empty((0, 21, 3), dtype=np.float32)
        else:
            landmarks = np.stack(sample_landmarks).astype(np.float32)

        return LandmarkExtractionResult(
            landmarks=landmarks,
            total_frames=total_frames,
            detected_frames=detected_frames,
        )

    def _extract_frame_landmarks(self, frame: Frame) -> np.ndarray | None:
        """
        Extract landmarks from one frame.

        OpenCV gives BGR frames.
        MediaPipe expects RGB/SRGB image data.
        """

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]

        landmarks = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
            dtype=np.float32,
        )

        return landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._landmarker.close()
