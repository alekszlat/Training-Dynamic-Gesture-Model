from pathlib import Path

import numpy as np

class LandmarkSaver:
    """Saves extracted landmark arrays to .npy files."""

    def __init__(self, landmarks_dir: Path) -> None:
        self.landmarks_dir = landmarks_dir

    def save(self, sample_id: str, landmarks: np.ndarray) -> Path:
        """
        Save landmarks for one sample.

        Args:
            sample_id: Unique sample ID from samples.csv.
            landmarks: NumPy array with shape [total_frames, 21, 3].

        Returns:
            Path to the saved .npy file.
        """

        self.landmarks_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.landmarks_dir / f"{sample_id}.npy"

        np.save(output_path, landmarks)

        return output_path