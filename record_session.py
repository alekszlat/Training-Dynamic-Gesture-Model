"""
Entry point for a webcam recording session.

Set LABEL and SPLIT below, then run:

    uv run python record_session.py

Author:
    Hristo Hristov
"""

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
    LABEL = Label.SWIPING_RIGHT
    SPLIT = Split.TRAIN

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
