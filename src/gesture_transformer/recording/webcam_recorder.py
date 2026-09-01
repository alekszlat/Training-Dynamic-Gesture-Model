"""
Records single gesture takes from a webcam to disk.

Dependencies:
    opencv-python: camera capture, preview window, mp4 writing.

Author:
    Hristo Hristov
"""

from datetime import datetime
from pathlib import Path

import cv2 as cv

from gesture_transformer.recording.config import Config

WINDOW_NAME = "Sample Recording"


class WebcamRecorder:
    """Records one take per call as an mp4 under the active split and label."""

    def __init__(self, config: Config):
        """
        Open the webcam.

        Args:
            config: Recording session settings.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """

        self.config = config
        self.recorder = cv.VideoCapture(config.webcam_id)

        if not self.recorder.isOpened():
            raise RuntimeError(
                f"Could not open webcam {config.webcam_id}. "
                "It may be in use by another program, or the id may be wrong."
            )

    def start_recording(self) -> Path:
        """
        Record one take until 'q' is pressed.

        The camera stays open so the next take does not pay to reopen it.
        Call close() when the session is over.

        Returns:
            Path to the written mp4.
        """

        fps = self._read_fps()
        width = int(self.recorder.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.recorder.get(cv.CAP_PROP_FRAME_HEIGHT))

        output_path = self._build_output_path()

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            self._capture_frames(writer)
        finally:
            writer.release()
            cv.destroyAllWindows()

        return output_path

    def wait_for_start(self) -> bool:
        """
        Show a live preview until the user starts the next take or ends the session.

        Returns:
            True to record the next take, False to end the session.
        """

        print("Space to record, 'e' to end the session.")

        # Created before the loop so waitKey has a window to read keys from even
        # while the camera is still warming up, and raised so it takes focus.
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)

        while True:
            success, frame = self.recorder.read()

            if success:
                cv.putText(
                    frame,
                    "space = record    e = end session",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv.imshow(WINDOW_NAME, frame)

            key = cv.waitKey(1)  # ms

            if key == ord(" "):
                return True

            if key == ord("e"):
                cv.destroyAllWindows()
                return False

    def close(self) -> None:
        """
        Release the camera. Safe to call more than once.

        Returns:
            None.
        """

        if self.recorder.isOpened():
            self.recorder.release()

    def _capture_frames(self, writer: cv.VideoWriter) -> None:
        """
        Write camera frames until the user quits or the camera stops responding.

        Args:
            writer: Open video writer for the current take.

        Returns:
            None.
        """

        consecutive_failures = 0

        while True:
            success, frame = self.recorder.read()

            if not success:
                consecutive_failures += 1

                if consecutive_failures >= self.config.max_consecutive_read_failures:
                    print("Camera stopped returning frames, ending take.")
                    break

                continue

            consecutive_failures = 0

            cv.imshow(WINDOW_NAME, frame)
            writer.write(frame)

            key = cv.waitKey(1)  # ms

            if key == ord("q"):
                break

    def _read_fps(self) -> float:
        """
        Report the camera frame rate, falling back to a default when it lies.

        A bad rate gives the file a broken timebase, which distorts the frame
        count the take is judged on.

        Returns:
            Frames per second to write the take at.
        """

        fps = self.recorder.get(cv.CAP_PROP_FPS)

        if not 1.0 <= fps <= self.config.max_fps:
            print(
                f"Camera reported fps={fps}, using {self.config.default_fps} instead."
            )
            return self.config.default_fps

        return fps

    def _build_output_path(self) -> Path:
        """
        Build a unique path for this take and create its folder.

        Returns:
            Path to write the take to.

        Raises:
            ValueError: If no active label is set.
        """

        if self.config.active_label is None:
            raise ValueError("Config.active_label must be set before recording.")

        label = self.config.active_label.value

        output_dir = self.config.output_dir / self.config.active_split.value / label
        output_dir.mkdir(parents=True, exist_ok=True)

        # Using a count is cleaner but difficult to maintain with deletions.
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")

        return output_dir / f"{label}_{timestamp}.mp4"
