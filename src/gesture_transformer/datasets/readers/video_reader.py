from collections.abc import Iterator
from pathlib import Path

import cv2 as cv
import numpy as np

from src.gesture_transformer.datasets.readers.base_reader import Frame


class VideoReader:
    """Reader that reads frames from a video file."""

    def read_frames(self, path: Path) -> Iterator[Frame]:
        """Read frames from the given video file path."""

        if not path.exists() or not path.is_file():
            raise ValueError(f"Invalid video file path: {path}")

        cap = cv.VideoCapture(str(path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if not isinstance(frame, np.ndarray):
                    raise TypeError(f"Invalid frame type: {type(frame)}")

                yield frame
        finally:
            cap.release()
