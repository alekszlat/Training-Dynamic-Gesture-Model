from collections.abc import Iterator
from pathlib import Path

import cv2 as cv

from src.gesture_transformer.datasets.readers.base_reader import Frame


class JesterFrameReader:
    """Reads frames from a Jester sample folder."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def read_frames(self, path: Path) -> Iterator[Frame]:
        if not path.exists():
            raise FileNotFoundError(f"Jester frame folder does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Expected a Jester frame folder, got: {path}")

        frame_paths = self._get_sorted_frame_paths(path)

        if not frame_paths:
            raise RuntimeError(f"No frame images found in folder: {path}")

        for frame_path in frame_paths:
            frame = cv.imread(str(frame_path))

            if frame is None:
                raise RuntimeError(f"Could not read frame image: {frame_path}")

            yield frame

    def _get_sorted_frame_paths(self, folder: Path) -> list[Path]:
        """
        Collect all valid frame image paths from a Jester sample folder
        and return them in the correct temporal order.

        Example Jester folder:
            videos/1/
                00001.jpg
                00002.jpg
                00003.jpg

        The order is important because this is a dynamic gesture.
        If the frames are not sorted correctly, the gesture motion becomes wrong.
        """

        # Go through every item inside the folder.
        # Keep only files with valid image extensions such as .jpg, .jpeg, or .png.
        frame_paths = [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in self.VALID_EXTENSIONS
        ]

        # Sort the frames using the numeric part of the filename.
        # This prevents incorrect ordering such as:
        # 00001.jpg, 00010.jpg, 00002.jpg
        return sorted(frame_paths, key=self._frame_sort_key)

    def _frame_sort_key(self, path: Path) -> int:
        """
        Convert a frame filename into an integer so frames can be sorted correctly.

        Example:
            00001.jpg -> 1
            00002.jpg -> 2
            00010.jpg -> 10

        This works because Jester frame filenames are numeric.
        """

        try:
            # path.stem returns the filename without extension.
            # Example:
            # Path("00001.jpg").stem -> "00001"
            #
            # int("00001") becomes 1.
            return int(path.stem)

        except ValueError as error:
            # If the filename is not numeric, sorting would be unreliable.
            # Example of invalid filename:
            # frame_a.jpg
            #
            # Raising an error here makes the dataset problem visible early.
            raise ValueError(
                f"Jester frame filename must be numeric, got: {path.name}"
            ) from error
