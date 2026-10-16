"""Unit tests for the recording tool.

Covers the session settings, the webcam recorder, the take validator, the
metadata appender and the session runner. The camera, the preview window and
MediaPipe are all stood in for, so a run needs no hardware and no model file.

A rejected take that still reaches the manifest, or a manifest a session
truncates instead of appends to, silently corrupts the dataset, so the
validator's thresholds and the appender's writes are covered hardest.

Author: Hristo Hristov
"""

import csv
import re
from dataclasses import fields
from pathlib import Path
from unittest.mock import call

import cv2 as real_cv2
import numpy as np
import pytest

from config import SUPPORTED_LABELS
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractionResult,
    LandmarkExtractor,
)
from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from gesture_transformer.datasets.readers.video_reader import VideoReader
from gesture_transformer.recording import (
    Label,
    MetadataAppender,
    RecorderConfig,
    SessionRunner,
    Split,
    TakeRecord,
    TakeValidationResult,
    TakeValidator,
    WebcamRecorder,
)
from gesture_transformer.recording.webcam_recorder import WINDOW_NAME

# ---------------------------------------------------------------------------
# Recording Config
# ---------------------------------------------------------------------------


def test_label_enum_has_one_member_per_supported_label():
    # arrange
    expected_count = len(SUPPORTED_LABELS)

    # act
    actual_count = len(Label)

    # assert
    assert actual_count == expected_count


def test_label_enum_order_matches_sorted_supported_labels():
    # arrange
    expected_names = [label.upper() for label in sorted(SUPPORTED_LABELS)]

    # act
    actual_names = list(Label.__members__)

    # assert
    assert actual_names == expected_names


@pytest.mark.parametrize("label", sorted(SUPPORTED_LABELS), ids=str)
def test_label_enum_maps_uppercase_name_to_original_label(label):
    # arrange
    member_name = label.upper()

    # act
    member = Label[member_name]

    # assert
    assert member.value == label


@pytest.mark.parametrize("member", Label, ids=lambda member: member.name)
def test_label_enum_names_are_uppercase(member):
    # arrange
    name = member.name

    # act
    uppercased = name.upper()

    # assert
    assert name == uppercased


@pytest.mark.parametrize(
    "split,expected_value",
    [
        (Split.TRAIN, "train"),
        (Split.TEST, "test"),
        (Split.VALIDATION, "validation"),
    ],
    ids=["train", "test", "validation"],
)
def test_split_value_is_serialized_name(split, expected_value):
    # arrange
    # (split and expected_value are provided by parametrize)

    # act
    value = split.value

    # assert
    assert value == expected_value


def test_new_recorder_config_defaults_to_train_split_and_no_active_label():
    # arrange
    # (no setup needed – testing the defaults)

    # act
    config = RecorderConfig()

    # assert
    assert (config.active_split, config.active_label) == (Split.TRAIN, None)


@pytest.mark.parametrize("label", Label, ids=lambda member: member.name)
def test_recorder_config_accepts_each_label(label):
    # arrange
    # (label is provided by parametrize)

    # act
    config = RecorderConfig(active_label=label)

    # assert
    assert config.active_label is label


@pytest.mark.parametrize("split", Split, ids=lambda split: split.name)
def test_recorder_config_accepts_each_split(split):
    # arrange
    # (split is provided by parametrize)

    # act
    config = RecorderConfig(active_split=split)

    # assert
    assert config.active_split is split


# ---------------------------------------------------------------------------
# Webcam Recorder
# ---------------------------------------------------------------------------


class FakeFrame:
    """Minimal frame that supports the .copy() the recorder calls on it."""

    def copy(self):
        return FakeFrame()


FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# What cv.waitKey returns when the user pressed nothing during the timeout.
NO_KEY = -1


class FakeCapture:
    """In-memory stand-in for cv.VideoCapture."""

    def __init__(
        self,
        *,
        opened=True,
        fps=30.0,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        frames=(),
    ):
        self._opened = opened
        self._fps = fps
        self._width = width
        self._height = height
        self._frames = list(frames)
        self.release_calls = 0
        self.read_calls = 0

    def isOpened(self):
        return self._opened

    def read(self):
        self.read_calls += 1
        if not self._frames:
            return False, None

        frame = self._frames.pop(0)

        # A None entry is a read the camera failed but recovered from.
        if frame is None:
            return False, None

        return True, frame

    def get(self, prop):
        return {
            real_cv2.CAP_PROP_FPS: self._fps,
            real_cv2.CAP_PROP_FRAME_WIDTH: self._width,
            real_cv2.CAP_PROP_FRAME_HEIGHT: self._height,
        }.get(prop, 0.0)

    def release(self):
        self.release_calls += 1
        self._opened = False


@pytest.fixture
def cv_mock(mocker):
    """Replace cv2 inside webcam_recorder, keeping the real enum constants."""
    # autospec so a call cv2 does not have, or with a signature it does not
    # accept, fails here instead of silently passing (U18).
    cv = mocker.patch(
        "gesture_transformer.recording.webcam_recorder.cv",
        autospec=True,
    )
    for name in (
        "CAP_PROP_FPS",
        "CAP_PROP_FRAME_WIDTH",
        "CAP_PROP_FRAME_HEIGHT",
        "EVENT_LBUTTONDOWN",
        "EVENT_MBUTTONDOWN",
        "WINDOW_NORMAL",
        "WND_PROP_TOPMOST",
        "FONT_HERSHEY_SIMPLEX",
    ):
        setattr(cv, name, getattr(real_cv2, name))

    return cv


@pytest.fixture
def active_label():
    """Any real Label member, for tests that need a non-None label."""
    return next(iter(Label))


def test_webcam_recorder_opens_the_camera_with_the_configured_id(cv_mock):
    # arrange
    CONFIGURED_WEBCAM_ID = 4
    cv_mock.VideoCapture.return_value = FakeCapture()
    config = RecorderConfig(webcam_id=CONFIGURED_WEBCAM_ID)

    # act
    WebcamRecorder(config)

    # assert
    cv_mock.VideoCapture.assert_called_once_with(CONFIGURED_WEBCAM_ID)


def test_webcam_recorder_raises_when_the_camera_cannot_be_opened(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(opened=False)
    config = RecorderConfig(webcam_id=7)

    # act / assert
    with pytest.raises(RuntimeError, match=r"webcam 7"):
        WebcamRecorder(config)


def test_close_releases_an_open_camera_only_once(cv_mock):
    # arrange
    camera = FakeCapture(opened=True)
    cv_mock.VideoCapture.return_value = camera
    recorder = WebcamRecorder(RecorderConfig())

    # act
    recorder.close()
    recorder.close()

    # assert
    assert camera.release_calls == 1


def test_wait_for_start_with_space_returns_true(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord(" ")
    recorder = WebcamRecorder(RecorderConfig())

    # act
    should_record = recorder.wait_for_start()

    # assert
    assert should_record is True


def test_wait_for_start_with_e_returns_false_and_closes_the_preview(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord("e")
    recorder = WebcamRecorder(RecorderConfig())

    # act
    should_record = recorder.wait_for_start()

    # assert
    assert should_record is False
    cv_mock.destroyAllWindows.assert_called_once()


def test_wait_for_start_with_left_click_returns_true(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    registered_callbacks = []
    cv_mock.setMouseCallback.side_effect = lambda _window, callback: (
        registered_callbacks.append(callback)
    )
    # The user clicks the preview the moment it is shown, and presses nothing.
    cv_mock.imshow.side_effect = lambda *_args: registered_callbacks[0](
        real_cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None
    )
    # Finite, so a recorder that ignored the click fails here instead of
    # spinning in the preview loop forever.
    cv_mock.waitKey.side_effect = [NO_KEY, NO_KEY, NO_KEY]
    recorder = WebcamRecorder(RecorderConfig())

    # act
    should_record = recorder.wait_for_start()

    # assert
    assert should_record is True


def test_wait_for_start_keeps_previewing_until_a_known_key(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(
        frames=[FakeFrame(), FakeFrame(), FakeFrame()]
    )
    cv_mock.waitKey.side_effect = [ord("x"), ord("y"), ord(" ")]
    recorder = WebcamRecorder(RecorderConfig())

    # act
    should_record = recorder.wait_for_start()

    # assert
    assert should_record is True


def test_wait_for_start_ignores_a_stale_start_click(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord("e")
    recorder = WebcamRecorder(RecorderConfig())
    recorder.start_clicked = True

    # act
    should_record = recorder.wait_for_start()

    # assert
    assert should_record is False


def test_wait_for_start_sets_up_the_preview_window(cv_mock):
    # arrange
    PREVIEW_SCALE = 2.0
    cv_mock.VideoCapture.return_value = FakeCapture()
    cv_mock.waitKey.return_value = ord("e")
    recorder = WebcamRecorder(RecorderConfig(preview_scale=PREVIEW_SCALE))

    # act
    recorder.wait_for_start()

    # assert
    cv_mock.resizeWindow.assert_called_once_with(
        WINDOW_NAME,
        int(FRAME_WIDTH * PREVIEW_SCALE),
        int(FRAME_HEIGHT * PREVIEW_SCALE),
    )


def test_start_recording_without_active_label_raises(cv_mock):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture()
    recorder = WebcamRecorder(RecorderConfig())  # active_label defaults to None

    # act / assert
    with pytest.raises(ValueError, match=r"active_label must be set"):
        recorder.start_recording()


def test_start_recording_writes_the_take_under_the_active_split_and_label(
    cv_mock, active_label, tmp_path
):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(
        active_label=active_label,
        active_split=Split.TEST,
        output_dir=tmp_path,
    )
    recorder = WebcamRecorder(config)

    # act
    output_path = recorder.start_recording()

    # assert
    assert output_path.parent == tmp_path / Split.TEST.value / active_label.value


def test_start_recording_names_the_take_after_its_label_and_a_timestamp(
    cv_mock, active_label, tmp_path
):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    output_path = recorder.start_recording()

    # assert
    # <label>_YYYYmmdd_HHMMSS_ffffff.mp4 - the timestamp is what keeps two
    # contributors' takes from overwriting each other, so its shape is part
    # of the contract.
    expected_name = rf"{re.escape(active_label.value)}_\d{{8}}_\d{{6}}_\d{{6}}\.mp4"

    assert re.fullmatch(expected_name, output_path.name)


@pytest.mark.parametrize(
    "reported_fps",
    [1.0, 30.0, 120.0],
    ids=["lower-boundary", "typical", "upper-boundary"],
)
def test_start_recording_uses_the_reported_fps_when_in_range(
    cv_mock, active_label, tmp_path, reported_fps
):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(
        fps=reported_fps, frames=[FakeFrame()]
    )
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    output_path = recorder.start_recording()

    # assert
    cv_mock.VideoWriter.assert_called_once_with(
        str(output_path),
        cv_mock.VideoWriter.fourcc.return_value,
        reported_fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )


@pytest.mark.parametrize(
    "reported_fps",
    [0.0, 0.5, 240.0, -1.0],
    ids=["zero", "below-min", "above-max", "negative"],
)
def test_start_recording_falls_back_to_default_fps_when_camera_lies(
    cv_mock, active_label, tmp_path, reported_fps
):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(
        fps=reported_fps, frames=[FakeFrame()]
    )
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    output_path = recorder.start_recording()

    # assert
    cv_mock.VideoWriter.assert_called_once_with(
        str(output_path),
        cv_mock.VideoWriter.fourcc.return_value,
        config.default_fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )


def test_start_recording_stops_on_middle_click(cv_mock, active_label, tmp_path):
    # arrange
    camera = FakeCapture(frames=[FakeFrame(), FakeFrame(), FakeFrame()])
    cv_mock.VideoCapture.return_value = camera
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # The session hands cv2 the mouse callback here; a take then has to honour
    # the clicks arriving through it, so the test drives that same callback
    # instead of reaching for the private handler.
    cv_mock.waitKey.return_value = ord(" ")
    recorder.wait_for_start()
    on_mouse = cv_mock.setMouseCallback.call_args.args[1]

    # What cv2 would do when the user middle-clicks the preview window.
    cv_mock.imshow.side_effect = lambda *_args: on_mouse(
        real_cv2.EVENT_MBUTTONDOWN, 0, 0, 0, None
    )
    # Finite, so a recorder that ignored the click fails here instead of
    # recording until the camera runs dry.
    cv_mock.waitKey.side_effect = [NO_KEY, NO_KEY, NO_KEY]
    frames_read_before_take = camera.read_calls

    # act
    recorder.start_recording()

    # assert
    assert camera.read_calls - frames_read_before_take == 1


def test_start_recording_stops_on_q(cv_mock, active_label, tmp_path):
    # arrange
    camera = FakeCapture(frames=[FakeFrame(), FakeFrame(), FakeFrame()])
    cv_mock.VideoCapture.return_value = camera
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    assert camera.read_calls == 1


def test_start_recording_writes_the_captured_frames_without_the_overlay(
    cv_mock, active_label, tmp_path
):
    # arrange
    frames = [FakeFrame(), FakeFrame()]
    cv_mock.VideoCapture.return_value = FakeCapture(frames=list(frames))
    cv_mock.waitKey.side_effect = [NO_KEY, ord("q")]
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    # The hints are drawn on a copy, so the saved take must hold the camera
    # frames themselves - a burnt-in legend would end up in the training data.
    writer = cv_mock.VideoWriter.return_value

    assert writer.write.call_args_list == [call(frames[0]), call(frames[1])]


def test_start_recording_ends_the_take_at_the_read_failure_limit(
    cv_mock, active_label, tmp_path
):
    # arrange
    MAX_READ_FAILURES = 3
    camera = FakeCapture(frames=[])
    cv_mock.VideoCapture.return_value = camera
    cv_mock.waitKey.return_value = NO_KEY
    config = RecorderConfig(
        active_label=active_label,
        output_dir=tmp_path,
        max_consecutive_read_failures=MAX_READ_FAILURES,
    )
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    assert camera.read_calls == MAX_READ_FAILURES


def test_start_recording_forgives_read_failures_a_good_frame_apart(
    cv_mock, active_label, tmp_path
):
    # arrange
    MAX_READ_FAILURES = 3
    # Two failures, a frame that resets the count, then enough to end the take.
    reads = [None, None, FakeFrame(), None, None, None]
    camera = FakeCapture(frames=reads)
    cv_mock.VideoCapture.return_value = camera
    cv_mock.waitKey.return_value = NO_KEY
    config = RecorderConfig(
        active_label=active_label,
        output_dir=tmp_path,
        max_consecutive_read_failures=MAX_READ_FAILURES,
    )
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    assert camera.read_calls == len(reads)


def test_start_recording_releases_the_writer(cv_mock, active_label, tmp_path):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    cv_mock.VideoWriter.return_value.release.assert_called_once_with()


def test_start_recording_releases_the_writer_when_the_take_blows_up(
    cv_mock, active_label, tmp_path
):
    # arrange
    cv_mock.VideoCapture.return_value = FakeCapture(frames=[FakeFrame()])
    cv_mock.imshow.side_effect = RuntimeError("no display")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    with pytest.raises(RuntimeError, match=r"no display"):
        recorder.start_recording()

    # assert
    # An unreleased writer leaves the mp4 header unwritten and the take unreadable.
    cv_mock.VideoWriter.return_value.release.assert_called_once_with()


def test_start_recording_leaves_the_camera_open_for_the_next_take(
    cv_mock, active_label, tmp_path
):
    # arrange
    camera = FakeCapture(frames=[FakeFrame()])
    cv_mock.VideoCapture.return_value = camera
    cv_mock.waitKey.return_value = ord("q")
    config = RecorderConfig(active_label=active_label, output_dir=tmp_path)
    recorder = WebcamRecorder(config)

    # act
    recorder.start_recording()

    # assert
    assert camera.release_calls == 0


# ---------------------------------------------------------------------------
# Take Validator
# ---------------------------------------------------------------------------


TAKE_PATH = Path("swiping_left_20260101_120000_000000.mp4")
MIN_FRAMES = 40
MIN_DETECTION_RATE = 0.5


def make_extraction_result(total_frames, detected_frames):
    """Extractor output with an array a test can recognise by identity."""
    return LandmarkExtractionResult(
        landmarks=np.zeros((total_frames, 21, 3), dtype=np.float32),
        total_frames=total_frames,
        detected_frames=detected_frames,
    )


@pytest.fixture
def video_reader(mocker):
    """Stand-in reader: the validator only forwards whatever it hands back."""
    reader = mocker.create_autospec(VideoReader, instance=True)
    reader.read_frames.return_value = iter(())
    return reader


@pytest.fixture
def landmark_extractor(mocker):
    """Stand-in extractor, so no MediaPipe model is loaded for a threshold test."""
    return mocker.create_autospec(LandmarkExtractor, instance=True)


@pytest.fixture
def validator_config():
    """Thresholds the validator tests are written against."""
    return RecorderConfig(
        min_frames=MIN_FRAMES,
        min_detection_rate=MIN_DETECTION_RATE,
    )


def test_validator_extracts_landmarks_from_the_frames_of_the_given_take(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    frames = iter(())
    video_reader.read_frames.return_value = frames
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=MIN_FRAMES, detected_frames=MIN_FRAMES
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    validator.validate(TAKE_PATH)

    # assert
    video_reader.read_frames.assert_called_once_with(TAKE_PATH)
    landmark_extractor.extract.assert_called_once_with(frames)


def test_validator_with_a_usable_take_returns_the_extracted_landmarks_and_counts(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    DETECTED_FRAMES = 30
    extraction_result = make_extraction_result(
        total_frames=MIN_FRAMES, detected_frames=DETECTED_FRAMES
    )
    landmark_extractor.extract.return_value = extraction_result
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result == TakeValidationResult(
        landmarks=extraction_result.landmarks,
        total_frames=MIN_FRAMES,
        detected_frames=DETECTED_FRAMES,
        detection_rate=0.75,
        is_valid=True,
        reason="",
    )


@pytest.mark.parametrize(
    "total_frames,expected_validity",
    [
        (MIN_FRAMES - 1, False),
        (MIN_FRAMES, True),
        (MIN_FRAMES + 1, True),
    ],
    ids=["one-below-min", "at-min", "one-above-min"],
)
def test_validator_keeps_a_take_from_min_frames_upwards(
    validator_config,
    landmark_extractor,
    video_reader,
    total_frames,
    expected_validity,
):
    # arrange
    # Every frame has a hand, so only the frame count can reject the take.
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=total_frames, detected_frames=total_frames
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.is_valid is expected_validity


@pytest.mark.parametrize(
    "detected_frames,expected_validity",
    [
        (49, False),
        (50, True),
        (51, True),
    ],
    ids=["one-below-min-rate", "at-min-rate", "one-above-min-rate"],
)
def test_validator_keeps_a_take_from_the_min_detection_rate_upwards(
    validator_config,
    landmark_extractor,
    video_reader,
    detected_frames,
    expected_validity,
):
    # arrange
    # 100 frames, so detected_frames reads as a percentage of the min rate.
    TOTAL_FRAMES = 100
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=TOTAL_FRAMES, detected_frames=detected_frames
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.is_valid is expected_validity


def test_validator_with_a_short_take_reports_the_frame_count_it_needed(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    RECORDED_FRAMES = MIN_FRAMES - 1
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=RECORDED_FRAMES, detected_frames=RECORDED_FRAMES
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.reason == f"only {RECORDED_FRAMES} frames, need {MIN_FRAMES}"


def test_validator_with_too_few_detections_reports_the_rate_it_needed(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    TOTAL_FRAMES = 100
    DETECTED_FRAMES = 49
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=TOTAL_FRAMES, detected_frames=DETECTED_FRAMES
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.reason == f"detection rate 0.49, need {MIN_DETECTION_RATE}"


def test_validator_with_a_take_failing_both_checks_reports_the_frame_count(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    # Too short and mostly undetected: the frame count is the one to report,
    # since a longer take is what the recorder has to do about it.
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=10, detected_frames=1
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.reason == f"only 10 frames, need {MIN_FRAMES}"


def test_validator_with_an_empty_take_reports_a_zero_detection_rate(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    # An unreadable take yields no frames; the rate must not divide by zero.
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=0, detected_frames=0
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.detection_rate == pytest.approx(0.0)


def test_validator_rounds_the_detection_rate_to_four_places(
    validator_config, landmark_extractor, video_reader
):
    # arrange
    # 1/3 is the shortest input with a rate that does not terminate.
    landmark_extractor.extract.return_value = make_extraction_result(
        total_frames=3, detected_frames=1
    )
    validator = TakeValidator(validator_config, landmark_extractor, video_reader)

    # act
    result = validator.validate(TAKE_PATH)

    # assert
    assert result.detection_rate == pytest.approx(0.3333)


# ---------------------------------------------------------------------------
# Metadata Appender
# ---------------------------------------------------------------------------


TAKE_FILENAME = "swiping_left_20260101_120000_000000.mp4"
TAKE_SAMPLE_ID = "recorded_swiping_left_20260101_120000_000000"
TAKE_TOTAL_FRAMES = 40
TAKE_DETECTED_FRAMES = 38
TAKE_DETECTION_RATE = 0.95


def make_take_record(path, **overrides):
    """A usable take, with only the fields a test is about overridden."""
    record_fields = {
        "label": "swiping_left",
        "landmarks": np.zeros((TAKE_TOTAL_FRAMES, 21, 3), dtype=np.float32),
        "total_frames": TAKE_TOTAL_FRAMES,
        "detected_frames": TAKE_DETECTED_FRAMES,
        "detection_rate": TAKE_DETECTION_RATE,
        "is_valid": True,
    }

    return TakeRecord(path=path, **{**record_fields, **overrides})


def read_rows(csv_path):
    """Rows of a csv as dicts, for whole-row assertions."""
    with open(csv_path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@pytest.fixture
def appender_config(tmp_path):
    """Settings pointing every written file inside this test's own directory."""
    return RecorderConfig(
        project_root=tmp_path,
        output_dir=tmp_path / "data" / "samples",
        manifest_path=tmp_path / "data" / "manifests" / "recorded_samples.csv",
        metadata_path=tmp_path / "data" / "processed" / "recorded_metadata.csv",
        landmarks_dir=tmp_path / "data" / "landmarks",
    )


@pytest.fixture
def landmark_saver(mocker, appender_config):
    """Saver that reports where it would have written, without touching disk."""
    saver = mocker.create_autospec(LandmarkSaver, instance=True)
    saver.save.side_effect = lambda sample_id, landmarks: (
        appender_config.landmarks_dir / f"{sample_id}.npy"
    )
    return saver


@pytest.fixture
def take_path(appender_config):
    """Where the recorder would have written a take of the default label."""
    return (
        appender_config.output_dir / Split.TRAIN.value / "swiping_left" / TAKE_FILENAME
    )


def test_appender_writes_one_manifest_row_per_usable_take(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    assert read_rows(appender_config.manifest_path) == [
        {
            "sample_id": TAKE_SAMPLE_ID,
            "source_type": "video",
            "source_name": "recorded",
            "external_id": "",
            "label": "swiping_left",
            "raw_label": "swiping_left",
            "path": f"data/samples/train/swiping_left/{TAKE_FILENAME}",
        }
    ]


def test_appender_writes_one_metadata_row_per_usable_take(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    assert read_rows(appender_config.metadata_path) == [
        {
            "sample_id": TAKE_SAMPLE_ID,
            "source_type": "video",
            "source_name": "recorded",
            "label": "swiping_left",
            "raw_label": "swiping_left",
            "path": f"data/samples/train/swiping_left/{TAKE_FILENAME}",
            "total_frames": str(TAKE_TOTAL_FRAMES),
            "detected_frames": str(TAKE_DETECTED_FRAMES),
            "detection_rate": str(TAKE_DETECTION_RATE),
            "status": "ok",
            "landmark_path": f"data/landmarks/{TAKE_SAMPLE_ID}.npy",
            "error": "",
        }
    ]


def test_appender_writes_the_metadata_columns_the_batch_pipeline_reads(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    # A recorded row has to load next to a batch-extracted one, so the columns
    # are the MetadataRecord fields, in order.
    with open(appender_config.metadata_path, encoding="utf-8", newline="") as file:
        header = csv.DictReader(file).fieldnames

    assert header == [field.name for field in fields(MetadataRecord)]


def test_appender_writes_the_canonical_label_beside_the_recorded_one(
    appender_config, landmark_saver, take_path
):
    # arrange
    # A label as a person would type it, which the manifest must not keep as is.
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path, label="Swiping Left")

    # act
    appender.append([record])

    # assert
    row = read_rows(appender_config.manifest_path)[0]

    assert (row["label"], row["raw_label"]) == ("swiping_left", "Swiping Left")


def test_appender_names_the_sample_after_the_takes_timestamped_file(
    appender_config, landmark_saver, take_path
):
    # arrange
    # Counting locally would restart at one per machine and two contributors
    # would overwrite each other's landmark files on merge.
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    assert read_rows(appender_config.manifest_path)[0]["sample_id"] == TAKE_SAMPLE_ID


def test_appender_writes_a_take_outside_the_project_as_an_absolute_path(
    appender_config, landmark_saver, tmp_path
):
    # arrange
    # Nothing to make the path relative to, so it stays usable as it is.
    outside_path = tmp_path.parent / "elsewhere" / TAKE_FILENAME
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(outside_path)

    # act
    appender.append([record])

    # assert
    assert read_rows(appender_config.manifest_path)[0]["path"] == (
        outside_path.as_posix()
    )


def test_appender_saves_the_landmarks_of_every_usable_take(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    landmark_saver.save.assert_called_once_with(
        sample_id=TAKE_SAMPLE_ID,
        landmarks=record.landmarks,
    )


def test_appender_leaves_rejected_takes_out_of_the_manifest(
    appender_config, landmark_saver, take_path
):
    # arrange
    rejected_path = take_path.with_name("swiping_left_20260101_130000_000000.mp4")
    appender = MetadataAppender(appender_config, landmark_saver)
    records = [
        make_take_record(take_path),
        make_take_record(rejected_path, is_valid=False),
    ]

    # act
    appender.append(records)

    # assert
    sample_ids = [row["sample_id"] for row in read_rows(appender_config.manifest_path)]

    assert sample_ids == [TAKE_SAMPLE_ID]


def test_appender_does_not_save_the_landmarks_of_a_rejected_take(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path, is_valid=False)

    # act
    appender.append([record])

    # assert
    landmark_saver.save.assert_not_called()


def test_appender_with_no_takes_writes_no_manifest(appender_config, landmark_saver):
    # arrange
    # An empty manifest is still a header-only file the batch pipeline reads.
    appender = MetadataAppender(appender_config, landmark_saver)

    # act
    appender.append([])

    # assert
    assert not appender_config.manifest_path.exists()


def test_appender_with_only_rejected_takes_writes_no_manifest(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path, is_valid=False)

    # act
    appender.append([record])

    # assert
    assert not appender_config.manifest_path.exists()


def test_appender_writes_a_header_for_a_new_manifest(
    appender_config, landmark_saver, take_path
):
    # arrange
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    header_line = appender_config.manifest_path.read_text(encoding="utf-8").splitlines()

    assert header_line[0] == ",".join(MetadataAppender.MANIFEST_FIELDNAMES)


def test_appender_keeps_the_rows_an_earlier_session_wrote(
    appender_config, landmark_saver, take_path
):
    # arrange
    EARLIER_SAMPLE_ID = "recorded_swiping_up_20251231_090000_000000"
    appender_config.manifest_path.parent.mkdir(parents=True)
    appender_config.manifest_path.write_text(
        ",".join(MetadataAppender.MANIFEST_FIELDNAMES)
        + f"\n{EARLIER_SAMPLE_ID},video,recorded,,swiping_up,swiping_up,earlier.mp4\n",
        encoding="utf-8",
    )
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    sample_ids = [row["sample_id"] for row in read_rows(appender_config.manifest_path)]

    assert sample_ids == [EARLIER_SAMPLE_ID, TAKE_SAMPLE_ID]


def test_appender_writes_no_second_header_into_an_existing_manifest(
    appender_config, landmark_saver, take_path
):
    # arrange
    header = ",".join(MetadataAppender.MANIFEST_FIELDNAMES)
    appender_config.manifest_path.parent.mkdir(parents=True)
    appender_config.manifest_path.write_text(f"{header}\n", encoding="utf-8")
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    # A second header row loads as a sample whose label is the word "label".
    lines = appender_config.manifest_path.read_text(encoding="utf-8").splitlines()

    assert lines.count(header) == 1


def test_appender_writes_every_take_of_a_session_in_order(
    appender_config, landmark_saver, take_path
):
    # arrange
    second_path = take_path.with_name("swiping_left_20260101_130000_000000.mp4")
    appender = MetadataAppender(appender_config, landmark_saver)
    records = [make_take_record(take_path), make_take_record(second_path)]

    # act
    appender.append(records)

    # assert
    sample_ids = [row["sample_id"] for row in read_rows(appender_config.manifest_path)]

    assert sample_ids == [TAKE_SAMPLE_ID, f"recorded_{second_path.stem}"]


def test_appender_creates_the_manifest_folder_when_it_is_missing(
    appender_config, landmark_saver, take_path
):
    # arrange
    # The first recorded session on a machine runs before data/manifests exists.
    appender = MetadataAppender(appender_config, landmark_saver)
    record = make_take_record(take_path)

    # act
    appender.append([record])

    # assert
    assert appender_config.manifest_path.is_file()


# ---------------------------------------------------------------------------
# Session Runner
# ---------------------------------------------------------------------------


def make_validation_result(**overrides):
    """The outcome of checking a take, with only the fields under test set."""
    result_fields = {
        "landmarks": np.zeros((TAKE_TOTAL_FRAMES, 21, 3), dtype=np.float32),
        "total_frames": TAKE_TOTAL_FRAMES,
        "detected_frames": TAKE_DETECTED_FRAMES,
        "detection_rate": TAKE_DETECTION_RATE,
        "is_valid": True,
        "reason": "",
    }

    return TakeValidationResult(**{**result_fields, **overrides})


@pytest.fixture
def session_config(active_label):
    """Settings of a session already set to a gesture."""
    return RecorderConfig(active_label=active_label)


@pytest.fixture
def recorder(mocker):
    """Stand-in for the recorder, so no camera or preview window is opened."""
    return mocker.create_autospec(WebcamRecorder, instance=True)


@pytest.fixture
def validator(mocker):
    """Stand-in checker, so the session's decisions can be driven directly."""
    return mocker.create_autospec(TakeValidator, instance=True)


@pytest.fixture
def metadata_appender(mocker):
    """Stand-in appender, so a session writes no manifest of its own."""
    return mocker.create_autospec(MetadataAppender, instance=True)


@pytest.fixture
def recorded_take(tmp_path):
    """A take on disk, as start_recording would leave it."""
    path = tmp_path / TAKE_FILENAME
    path.touch()
    return path


def test_run_records_a_take_for_every_start(
    session_config, recorder, validator, metadata_appender, tmp_path
):
    # arrange
    first_take = tmp_path / "take_one.mp4"
    second_take = tmp_path / "take_two.mp4"
    first_take.touch()
    second_take.touch()
    recorder.wait_for_start.side_effect = [True, True, False]
    recorder.start_recording.side_effect = [first_take, second_take]
    validator.validate.return_value = make_validation_result()
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    assert validator.validate.call_args_list == [call(first_take), call(second_take)]


def test_run_ends_without_recording_when_the_user_declines_the_first_take(
    session_config, recorder, validator, metadata_appender
):
    # arrange
    recorder.wait_for_start.return_value = False
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    recorder.start_recording.assert_not_called()


def test_run_deletes_a_rejected_take(
    session_config, recorder, validator, metadata_appender, recorded_take
):
    # arrange
    # A rejected take never reaches the manifest, so the file is dead weight.
    recorder.wait_for_start.side_effect = [True, False]
    recorder.start_recording.return_value = recorded_take
    validator.validate.return_value = make_validation_result(
        is_valid=False, reason="only 10 frames, need 40"
    )
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    assert not recorded_take.exists()


def test_run_keeps_a_usable_take(
    session_config, recorder, validator, metadata_appender, recorded_take
):
    # arrange
    recorder.wait_for_start.side_effect = [True, False]
    recorder.start_recording.return_value = recorded_take
    validator.validate.return_value = make_validation_result()
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    assert recorded_take.is_file()


def test_run_appends_the_takes_of_the_whole_session_at_the_end(
    session_config, recorder, validator, metadata_appender, recorded_take
):
    # arrange
    # Rejected takes go to the appender too; leaving them out is its job.
    result = make_validation_result(is_valid=False, reason="only 10 frames, need 40")
    recorder.wait_for_start.side_effect = [True, False]
    recorder.start_recording.return_value = recorded_take
    validator.validate.return_value = result
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    metadata_appender.append.assert_called_once_with(
        [
            TakeRecord(
                path=recorded_take,
                label=session_config.active_label.value,
                landmarks=result.landmarks,
                total_frames=result.total_frames,
                detected_frames=result.detected_frames,
                detection_rate=result.detection_rate,
                is_valid=result.is_valid,
            )
        ]
    )


def test_run_closes_the_camera_when_the_session_ends(
    session_config, recorder, validator, metadata_appender
):
    # arrange
    recorder.wait_for_start.return_value = False
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    runner.run()

    # assert
    recorder.close.assert_called_once_with()


def test_run_closes_the_camera_when_a_take_blows_up(
    session_config, recorder, validator, metadata_appender
):
    # arrange
    # Without the release the camera stays claimed until the shell is closed.
    recorder.wait_for_start.return_value = True
    recorder.start_recording.side_effect = RuntimeError("camera unplugged")
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    with pytest.raises(RuntimeError, match=r"camera unplugged"):
        runner.run()

    # assert
    recorder.close.assert_called_once_with()


def test_run_appends_the_takes_it_already_had_when_a_take_blows_up(
    session_config, recorder, validator, metadata_appender, recorded_take
):
    # arrange
    # The first take is worth keeping even though the second one killed the run.
    recorder.wait_for_start.return_value = True
    recorder.start_recording.side_effect = [recorded_take, RuntimeError("unplugged")]
    validator.validate.return_value = make_validation_result()
    runner = SessionRunner(session_config, recorder, validator, metadata_appender)

    # act
    with pytest.raises(RuntimeError, match=r"unplugged"):
        runner.run()

    # assert
    appended_paths = [
        record.path for record in metadata_appender.append.call_args.args[0]
    ]

    assert appended_paths == [recorded_take]


def test_run_without_an_active_label_raises(recorder, validator, metadata_appender):
    # arrange
    recorder.wait_for_start.return_value = True
    config = RecorderConfig()  # active_label defaults to None
    runner = SessionRunner(config, recorder, validator, metadata_appender)

    # act / assert
    with pytest.raises(ValueError, match=r"active_label must be set"):
        runner.run()
