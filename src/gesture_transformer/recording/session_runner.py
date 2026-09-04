"""
Drives a recording session: record a take, check it, report it, append the lot at the end.

Dependencies:
    gesture_transformer.recording: webcam recorder, take validator, settings.

Author:
    Hristo Hristov
"""

from pathlib import Path

from gesture_transformer.recording.metadata_appender import (
    MetadataAppender,
    TakeRecord,
)
from gesture_transformer.recording.recorder_config import RecorderConfig
from gesture_transformer.recording.take_validator import (
    TakeValidationResult,
    TakeValidator,
)
from gesture_transformer.recording.webcam_recorder import WebcamRecorder


class SessionRunner:
    """Records takes until the user ends the session, then appends them to the manifest."""

    def __init__(
        self,
        recorder_config: RecorderConfig,
        recorder: WebcamRecorder,
        validator: TakeValidator,
        metadata_appender: MetadataAppender,
    ):
        """
        Store the settings and the collaborators used during a session.

        Args:
            recorder_config: Recording session settings.
            recorder: Recorder owning the camera and the preview window.
            validator: Checker run over each finished take.
            metadata_appender: Appends the session's takes to the manifest.
        """

        self.recorder_config = recorder_config
        self.recorder = recorder
        self.validator = validator
        self.metadata_appender = metadata_appender
        self.records: list[TakeRecord] = []

    def run(self) -> None:
        """
        Record takes until the user ends the session.

        Records are kept in memory and appended in one go, so a session adds to
        the existing files instead of rewriting them.

        Returns:
            None.
        """

        try:
            while self.recorder.wait_for_start():
                self._record_take()
        finally:
            self.recorder.close()
            self.metadata_appender.append(self.records)

    def _record_take(self) -> None:
        """
        Record one take, check it and report the outcome. Rejected takes are deleted.

        Returns:
            None.
        """

        label = self.recorder_config.active_label

        if label is None:
            raise ValueError(
                "recorder_config.active_label must be set before recording."
            )

        take_path = self.recorder.start_recording()
        result = self.validator.validate(take_path)

        # A rejected take never reaches the manifest, so the file has no use.
        if not result.is_valid:
            take_path.unlink()

        self.records.append(
            TakeRecord(
                path=take_path,
                label=label.value,
                landmarks=result.landmarks,
                total_frames=result.total_frames,
                detected_frames=result.detected_frames,
                detection_rate=result.detection_rate,
                is_valid=result.is_valid,
            )
        )

        self._print_result(take_path, result)

    def _print_result(self, take_path: Path, result: TakeValidationResult) -> None:
        """
        Print how one take scored and how many the session has kept so far.

        Args:
            take_path: Path to the recorded mp4.
            result: Outcome of checking the take.

        Returns:
            None.
        """

        kept = sum(1 for record in self.records if record.is_valid)
        status = "kept" if result.is_valid else f"rejected and deleted, {result.reason}"

        print(f"{take_path.name}: {status}")
        print(
            f"  frames={result.total_frames}, "
            f"detected={result.detected_frames}, "
            f"rate={result.detection_rate}"
        )
        print(f"  kept {kept} of {len(self.records)} takes this session")
