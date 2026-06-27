from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Final

import cv2 as cv
import mediapipe as mp
import numpy as np


class LandmarkExtractor:
    """Extract one-hand MediaPipe landmarks from a single image frame.

    Output shape:
        21 landmarks × 3 coordinates = 63 features

    This extractor only returns raw x, y, z landmark coordinates.
    Movement deltas such as Δx, Δy, Δz are handled later when we process
    a sequence of frames.
    """

    LANDMARK_COUNT: Final[int] = 21
    COORDINATE_COUNT: Final[int] = 3
    FEATURE_COUNT: Final[int] = LANDMARK_COUNT * COORDINATE_COUNT

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        flip_horizontal: bool = False,
        model_asset_path: str | Path | None = None,
    ) -> None:
        if max_num_hands != 1:
            raise ValueError("LandmarkExtractor currently supports only max_num_hands=1")

        self.max_num_hands = max_num_hands
        self.flip_horizontal = flip_horizontal
        self.model_asset_path = Path(model_asset_path) if model_asset_path else None
        self._landmarker: Any | None = None

    def extract(self, frame: np.ndarray) -> list[float]:
        """Extract one flat 63-value feature vector from one image frame."""
        self._validate_frame(frame)

        processed_frame = frame

        if self.flip_horizontal:
            processed_frame = cv.flip(processed_frame, 1)

        # Useful for tests:
        # A completely blank image cannot contain a hand, so we can return zeros
        # without requiring a MediaPipe model file.
        if np.count_nonzero(processed_frame) == 0:
            return self._empty_features()

        rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
        landmarks = self._detect_first_hand(rgb_frame)

        if landmarks is None:
            return self._empty_features()

        features = self._flatten_landmarks(landmarks)

        if len(features) != self.FEATURE_COUNT:
            raise RuntimeError(
                f"expected {self.FEATURE_COUNT} features, got {len(features)}"
            )

        return features

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> LandmarkExtractor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @classmethod
    def _empty_features(cls) -> list[float]:
        return [0.0] * cls.FEATURE_COUNT

    @classmethod
    def _flatten_landmarks(cls, landmarks: list[Any]) -> list[float]:
        features: list[float] = []

        for landmark in landmarks:
            features.extend(
                [
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z),
                ]
            )

        return features

    def _detect_first_hand(self, rgb_frame: np.ndarray) -> list[Any] | None:
        landmarker = self._get_landmarker()

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = landmarker.detect(image)

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks[0]

    def _get_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker

        if self.model_asset_path is None:
            raise RuntimeError(
                "MediaPipe hand detection requires model_asset_path. "
                "Provide the path to hand_landmarker.task."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_asset_path)
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
        )

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self._landmarker

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("frame must not be empty")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape H x W x 3")

        if frame.dtype != np.uint8:
            raise ValueError("frame must use dtype np.uint8")
