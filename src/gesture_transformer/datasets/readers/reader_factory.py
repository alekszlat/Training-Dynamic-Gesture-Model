from src.gesture_transformer.datasets.readers.jester_frame_reader import JesterFrameReader
from src.gesture_transformer.datasets.readers.video_reader import VideoReader
from src.gesture_transformer.datasets.readers.base_reader import FrameReader

class ReaderFactory:
    """Factory class to create frame readers based on the data source type."""

    def __init__(self) -> None:
        self._readers: dict[str, FrameReader] = {
            "jester": JesterFrameReader(),
            "video": VideoReader(),
        }

    def get_reader(self, source_type: str) -> FrameReader:
        """Return a frame reader instance based on the source type."""
       
        try:
            return self._readers[source_type]
        except KeyError:
            available = ", ".join(self._readers.keys())
            raise ValueError(
                f"Unknown source_type: {source_type}. "
                f"Available source types: {available}"
            )