from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, asdict
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
    """Result of running MediaPipe over one gesture sample."""

    landmarks: np.ndarray
    total_frames: int
    detected_frames: int


@dataclass(frozen=True)
class LandmarkMetadata:
    """One row for metadata.csv after landmark extraction."""

    sample_id: str
    source_type: str
    source_name: str
    label: str
    raw_label: str
    path: str
    total_frames: int
    detected_frames: int
    detection_rate: float
    status: str
    landmark_path: str
    error: str

    def to_csv_row(self) -> dict[str, object]:
        return asdict(self)


class LandmarkExtractor:
    """Extracts one-hand MediaPipe landmarks from a sequence of frames."""

    def __init__(
        self,
        model_path: str | Path = "hand_landmarker.task",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_detection_rate: float = 0.7,
    ) -> None:
        self.min_detection_rate = min_detection_rate

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = HandLandmarker.create_from_options(options)

    def extract_and_save(
        self,
        *,
        frames: Iterable[Frame],
        sample_id: str,
        source_type: str,
        source_name: str,
        label: str,
        raw_label: str,
        sample_path: Path,
        landmarks_dir: Path,
    ) -> LandmarkMetadata:
        """
        Extract landmarks for one sample, save valid landmarks as .npy,
        and return one metadata row.

        Saved landmark file:
            data/interim/landmarks/{sample_id}.npy

        Landmark shape:
            [num_detected_frames, 21, 3]
        """

        try:
            result = self.extract(frames)
            status, error = self._get_status(result)

            landmark_path = ""

            if status == "ok":
                landmarks_dir.mkdir(parents=True, exist_ok=True)

                output_path = landmarks_dir / f"{sample_id}.npy"
                np.save(output_path, result.landmarks)

                landmark_path = str(output_path)

            return LandmarkMetadata(
                sample_id=sample_id,
                source_type=source_type,
                source_name=source_name,
                label=label,
                raw_label=raw_label,
                path=str(sample_path),
                total_frames=result.total_frames,
                detected_frames=result.detected_frames,
                detection_rate=self._calculate_detection_rate(
                    result.total_frames,
                    result.detected_frames,
                ),
                status=status,
                landmark_path=landmark_path,
                error=error,
            )

        except Exception as error:
            return LandmarkMetadata(
                sample_id=sample_id,
                source_type=source_type,
                source_name=source_name,
                label=label,
                raw_label=raw_label,
                path=str(sample_path),
                total_frames=0,
                detected_frames=0,
                detection_rate=0.0,
                status="error",
                landmark_path="",
                error=str(error),
            )

    def extract(self, frames: Iterable[Frame]) -> LandmarkExtractionResult:
        """
        Extract landmarks from all frames in one gesture sample.

        Returns:
            landmarks with shape [num_detected_frames, 21, 3]

        Frames where no hand is detected are skipped.
        """

        sample_landmarks: list[np.ndarray] = []
        total_frames = 0

        for frame in frames:
            total_frames += 1

            frame_landmarks = self._extract_frame_landmarks(frame)

            if frame_landmarks is None:
                continue

            sample_landmarks.append(frame_landmarks)

        if not sample_landmarks:
            landmarks = np.empty((0, 21, 3), dtype=np.float32)
        else:
            landmarks = np.stack(sample_landmarks).astype(np.float32)

        return LandmarkExtractionResult(
            landmarks=landmarks,
            total_frames=total_frames,
            detected_frames=len(sample_landmarks),
        )

    def _extract_frame_landmarks(self, frame: Frame) -> np.ndarray | None:
        """
        Extract landmarks from one frame.

        OpenCV returns BGR frames.
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
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in hand_landmarks
            ],
            dtype=np.float32,
        )

        return landmarks

    def _get_status(self, result: LandmarkExtractionResult) -> tuple[str, str]:
        """Decide whether the extracted sample is usable."""

        if result.total_frames == 0:
            return "error", "no_frames_found"

        if result.detected_frames == 0:
            return "rejected_no_landmarks", "no_hand_landmarks_detected"

        detection_rate = self._calculate_detection_rate(
            result.total_frames,
            result.detected_frames,
        )

        if detection_rate < self.min_detection_rate:
            return "rejected_low_detection_rate", "too_few_detected_frames"

        return "ok", ""

    def _calculate_detection_rate(
        self,
        total_frames: int,
        detected_frames: int,
    ) -> float:
        """Calculate detected_frames / total_frames safely."""

        if total_frames == 0:
            return 0.0

        return round(detected_frames / total_frames, 4)

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._landmarker.close()