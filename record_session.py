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
    Config,
    Label,
    MetadataAppender,
    SessionRunner,
    Split,
    TakeValidator,
    WebcamRecorder,
)

if __name__ == "__main__":
    LABEL = Label.CLICK
    SPLIT = Split.TRAIN

    config = Config(active_label=LABEL, active_split=SPLIT)

    landmark_extractor = LandmarkExtractor(model_path=config.model_path)

    runner = SessionRunner(
        config=config,
        recorder=WebcamRecorder(config),
        validator=TakeValidator(
            config=config,
            landmark_extractor=landmark_extractor,
            video_reader=VideoReader(),
        ),
        metadata_appender=MetadataAppender(
            config=config,
            landmark_saver=LandmarkSaver(landmarks_dir=config.landmarks_dir),
        ),
    )

    try:
        runner.run()
    finally:
        landmark_extractor.close()
