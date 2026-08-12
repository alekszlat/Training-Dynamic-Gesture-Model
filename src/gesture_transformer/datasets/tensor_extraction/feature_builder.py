import numpy as np


class FeatureBuilder:
    """Builds Transformer-ready features from normalized hand landmarks."""

    def build(self, landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Convert normalized landmarks into model features.

        Args:
            landmarks: [40, 21, 3]
            mask: [40]

        Returns:
            features: [40, 126]
        """

        self._validate_input(landmarks, mask)

        relative = self._to_wrist_relative(landmarks)

        delta = self._calculate_deltas(relative)

        combined = self._combine_relative_and_delta(relative, delta)

        features = self._flatten_features(combined)

        features = self._apply_mask(features, mask)

        return features.astype(np.float32)

    def _validate_input(self, landmarks: np.ndarray, mask: np.ndarray) -> None:
        """Check that the input shapes are correct."""

        if landmarks.ndim != 3:
            raise ValueError(
                f"Expected landmarks to have 3 dimensions, got {landmarks.shape}"
            )

        if landmarks.shape[1:] != (21, 3):
            raise ValueError(
                f"Expected landmarks shape [T, 21, 3], got {landmarks.shape}"
            )

        if mask.ndim != 1:
            raise ValueError(f"Expected mask shape [T], got {mask.shape}")

        if landmarks.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Landmarks and mask length mismatch: "
                f"{landmarks.shape[0]} != {mask.shape[0]}"
            )

    def _to_wrist_relative(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert absolute MediaPipe coordinates to wrist-relative coordinates.

        Input:
            landmarks: [40, 21, 3]

        Output:
            relative: [40, 21, 3]
        """

        wrist = landmarks[:, 0:1, :]

        relative = landmarks - wrist

        return relative

    def _calculate_deltas(self, relative: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame movement.

        Input:
            relative: [40, 21, 3]

        Output:
            delta: [40, 21, 3]
        """

        delta = np.zeros_like(relative, dtype=np.float32)

        delta[1:] = relative[1:] - relative[:-1]

        return delta

    def _combine_relative_and_delta(
        self,
        relative: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """
        Combine position features and movement features.

        Input:
            relative: [40, 21, 3]
            delta:    [40, 21, 3]

        Output:
            combined: [40, 21, 6]
        """

        combined = np.concatenate([relative, delta], axis=-1)

        return combined

    def _flatten_features(self, combined: np.ndarray) -> np.ndarray:
        """
        Flatten landmarks into one feature vector per frame.

        Input:
            combined: [40, 21, 6]

        Output:
            features: [40, 126]
        """

        num_frames = combined.shape[0]

        features = combined.reshape(num_frames, -1)

        return features

    def _apply_mask(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Zero out padded frames.

        Input:
            features: [40, 126]
            mask:     [40]

        Output:
            masked features: [40, 126]
        """

        return features * mask[:, None]
