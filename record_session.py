"""
Entry point for a webcam recording session.

Set SPLIT, and TEST_USER for the test split, below, then run:

    uv run python record_session.py

LABEL is only the gesture the session starts on. Click a gesture in the preview
window to switch it between takes.

Author:
    Hristo Hristov
"""

from config import LANDMARK_OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractor,
)
from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.readers.video_reader import VideoReader
from gesture_transformer.recording import (
    Label,
    MetadataAppender,
    RecorderConfig,
    SessionRunner,
    Split,
    TakeValidator,
    WebcamRecorder,
)

if __name__ == "__main__":
    LABEL = Label.DOING_OTHER_THINGS
    SPLIT = Split.TEST
    TEST_USER = 5

    recorder_config = None

    if SPLIT == Split.TEST:
        recorder_config = RecorderConfig(
            active_label=LABEL,
            active_split=SPLIT,
            metadata_path=PROCESSED_DIR / "test" / f"user_{TEST_USER}_metadata.csv",
            manifest_path=PROJECT_ROOT
            / "data"
            / "manifests"
            / "test"
            / f"user_{TEST_USER}_manifest.csv",
            landmarks_dir=LANDMARK_OUTPUT_DIR / "test" / f"user_{TEST_USER}",
        )
    else:
        recorder_config = RecorderConfig(active_label=LABEL, active_split=SPLIT)

    landmark_extractor = LandmarkExtractor(
        model_path=recorder_config.model_path, delegate=recorder_config.delegate
    )

    runner = SessionRunner(
        recorder_config=recorder_config,
        recorder=WebcamRecorder(recorder_config),
        validator=TakeValidator(
            recorder_config=recorder_config,
            landmark_extractor=landmark_extractor,
            video_reader=VideoReader(),
        ),
        metadata_appender=MetadataAppender(
            recorder_config=recorder_config,
            landmark_saver=LandmarkSaver(landmarks_dir=recorder_config.landmarks_dir),
        ),
    )

    try:
        runner.run()
    finally:
        landmark_extractor.close()
