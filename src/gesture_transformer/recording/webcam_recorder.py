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

from gesture_transformer.recording.recorder_config import Label, RecorderConfig

WINDOW_NAME = "Sample Recording"

IDLE_HINTS = [
    "space / left click  = start take",
    "q / middle click    = stop take",
    "e                   = end session",
    "click a gesture     = switch label",
]

RECORDING_HINTS = [
    "RECORDING",
    "q / middle click = stop take",
]

NO_LABEL_HINT = "pick a gesture to start"

# The buttons start below the last idle hint, which _draw_hints puts at y=108.
# BUTTON_WIDTH fits doing_other_things, the longest label, at the button font size.
BUTTON_LEFT = 10
BUTTON_TOP = 126
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 32
BUTTON_GAP = 6

ACTIVE_BUTTON_COLOUR = (0, 160, 0)
INACTIVE_BUTTON_COLOUR = (60, 60, 60)
BUTTON_TEXT_COLOUR = (255, 255, 255)
NO_LABEL_COLOUR = (0, 255, 255)

# One (label, (x, y, width, height)) entry per label, top to bottom, in frame pixels.
LABEL_BUTTONS = [
    (
        label,
        (
            BUTTON_LEFT,
            BUTTON_TOP + index * (BUTTON_HEIGHT + BUTTON_GAP),
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
    )
    for index, label in enumerate(Label)
]


class WebcamRecorder:
    """Records one take per call as an mp4 under the active split and label."""

    def __init__(self, recorder_config: RecorderConfig):
        """
        Open the webcam.

        Args:
            recorder_config: Recording session settings.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """

        self.recorder_config = recorder_config
        self.recorder = cv.VideoCapture(recorder_config.webcam_id)
        self.start_clicked = False
        self.stop_clicked = False
        self.is_idle = False

        if not self.recorder.isOpened():
            raise RuntimeError(
                f"Could not open webcam {recorder_config.webcam_id}. "
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

        fourcc = cv.VideoWriter.fourcc(*"mp4v")
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

        The preview shows one button per label, and clicking a button switches the
        active label. A start is ignored while no label is active.

        Returns:
            True to record the next take, False to end the session.
        """

        print("Space to record, 'e' to end, click a gesture to switch label.")

        # Created before the loop so waitKey has a window to read keys from even
        # while the camera is still warming up, and raised so it takes focus.
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.resizeWindow(WINDOW_NAME, *self._preview_size())
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
        cv.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self.start_clicked = False
        self.is_idle = True

        while True:
            success, frame = self.recorder.read()

            if success:
                self._draw_hints(frame, IDLE_HINTS, (0, 255, 0))
                self._draw_label_buttons(frame)
                cv.imshow(WINDOW_NAME, frame)

            key = cv.waitKey(1)  # ms

            if key == ord(" ") or self.start_clicked:
                if self.recorder_config.active_label is not None:
                    self.is_idle = False
                    return True

                # If the ignored start click stayed set, the next click on a
                # label button would start a take immediately.
                self.start_clicked = False

            if key == ord("e"):
                self.is_idle = False
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
        self.stop_clicked = False

        while True:
            success, frame = self.recorder.read()

            if not success:
                consecutive_failures += 1

                if (
                    consecutive_failures
                    >= self.recorder_config.max_consecutive_read_failures
                ):
                    print("Camera stopped returning frames, ending take.")
                    break

                continue

            consecutive_failures = 0

            writer.write(frame)

            preview = frame.copy()
            self._draw_hints(preview, RECORDING_HINTS, (0, 0, 255))
            cv.imshow(WINDOW_NAME, preview)

            key = cv.waitKey(1)  # ms

            if key == ord("q") or self.stop_clicked:
                break

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """
        Record clicks on the preview window so takes can run without the keyboard.

        Using the mouse leaves the recording hand free to change body position and
        distance between takes. A left click on a label button switches the label
        instead of starting a take. Label buttons work only between takes, so the
        label cannot change while a take records.

        Args:
            event: OpenCV mouse event code.
            x: Cursor column, in frame pixels.
            y: Cursor row, in frame pixels.
            flags: Event flags. Unused.
            param: Callback data. Unused.

        Returns:
            None.
        """

        if event == cv.EVENT_LBUTTONDOWN:
            clicked_label = self._label_at(x, y) if self.is_idle else None

            if clicked_label is not None:
                self.recorder_config.active_label = clicked_label
            else:
                self.start_clicked = True
        elif event == cv.EVENT_MBUTTONDOWN:
            self.stop_clicked = True

    def _label_at(self, x: int, y: int) -> Label | None:
        """
        Find the label button under a point on the preview.

        Args:
            x: Column, in frame pixels.
            y: Row, in frame pixels.

        Returns:
            The label of the button under the point, or None outside every button.
        """

        for label, (left, top, width, height) in LABEL_BUTTONS:
            if left <= x < left + width and top <= y < top + height:
                return label

        return None

    def _draw_label_buttons(self, frame) -> None:
        """
        Draw one button per label onto a preview frame, the active one highlighted.

        Args:
            frame: Frame to draw on, modified in place.

        Returns:
            None.
        """

        active_label = self.recorder_config.active_label

        for label, (left, top, width, height) in LABEL_BUTTONS:
            colour = (
                ACTIVE_BUTTON_COLOUR
                if label is active_label
                else INACTIVE_BUTTON_COLOUR
            )
            cv.rectangle(
                frame, (left, top), (left + width, top + height), colour, cv.FILLED
            )
            cv.putText(
                frame,
                label.value,
                (left + 8, top + 22),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                BUTTON_TEXT_COLOUR,
                1,
            )

        if active_label is None:
            _, last_top, _, last_height = LABEL_BUTTONS[-1][1]
            cv.putText(
                frame,
                NO_LABEL_HINT,
                (BUTTON_LEFT, last_top + last_height + 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                NO_LABEL_COLOUR,
                2,
            )

    def _draw_hints(self, frame, lines: list[str], colour: tuple) -> None:
        """
        Draw the control legend onto a preview frame.

        Args:
            frame: Frame to draw on, modified in place.
            lines: Legend lines, top to bottom.
            colour: BGR colour for the text.

        Returns:
            None.
        """

        for index, line in enumerate(lines):
            cv.putText(
                frame,
                line,
                (10, 30 + index * 26),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

    def _preview_size(self) -> tuple[int, int]:
        """
        Scale the camera frame size up for the preview window.

        Returns:
            Preview width and height in pixels.
        """

        scale = self.recorder_config.preview_scale
        width = self.recorder.get(cv.CAP_PROP_FRAME_WIDTH) * scale
        height = self.recorder.get(cv.CAP_PROP_FRAME_HEIGHT) * scale

        return int(width), int(height)

    def _read_fps(self) -> float:
        """
        Report the camera frame rate, falling back to a default when it lies.

        A bad rate gives the file a broken timebase, which distorts the frame
        count the take is judged on.

        Returns:
            Frames per second to write the take at.
        """

        fps = self.recorder.get(cv.CAP_PROP_FPS)

        if not 1.0 <= fps <= self.recorder_config.max_fps:
            print(
                f"Camera reported fps={fps}, using {self.recorder_config.default_fps} instead."
            )
            return self.recorder_config.default_fps

        return fps

    def _build_output_path(self) -> Path:
        """
        Build a unique path for this take and create its folder.

        Returns:
            Path to write the take to.

        Raises:
            ValueError: If no active label is set.
        """

        if self.recorder_config.active_label is None:
            raise ValueError(
                "recorder_config.active_label must be set before recording."
            )

        label = self.recorder_config.active_label.value

        output_dir = (
            self.recorder_config.output_dir
            / self.recorder_config.active_split.value
            / label
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Using a count is cleaner but difficult to maintain with deletions.
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")

        return output_dir / f"{label}_{timestamp}.mp4"
