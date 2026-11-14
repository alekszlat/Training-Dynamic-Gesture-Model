from pathlib import Path

import numpy as np


class LandmarkSaver:
    """Saves extracted landmark arrays to .npy files."""

    def __init__(self, landmarks_dir: Path) -> None:
        self.landmarks_dir = landmarks_dir

    def save(self, sample_id: str, landmarks: np.ndarray, subdir: str = "") -> Path:
        """
        Save landmarks for one sample.

        Args:
            sample_id: Unique sample ID from samples.csv.
            landmarks: NumPy array with shape [total_frames, 21, 3].
            subdir: Folder under landmarks_dir, for example the source name.
                Empty saves directly in landmarks_dir.

        Returns:
            Path to the saved .npy file.
        """

        output_dir = self.landmarks_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{sample_id}.npy"

        np.save(output_path, landmarks)

        return output_path
