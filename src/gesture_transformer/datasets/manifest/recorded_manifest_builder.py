from pathlib import Path
from typing import ClassVar

from src.gesture_transformer.datasets.manifest.label_mapper import LabelMapper


class RecordedManifestBuilder:
    """Build manifest rows for personally recorded gesture videos."""

    VALID_VIDEO_EXTENSIONS: ClassVar[set[str]] = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        samples_dir: Path,
        project_root: Path | None = None,
    ) -> None:
        self.samples_dir = samples_dir
        self.project_root = project_root
        self.label_mapper = LabelMapper()

    def build(self) -> list[dict[str, str]]:
        """
        Build manifest rows for recorded gesture samples.

        Expected folder structure:

            data/raw/recorded/
              swipe_up/
                001.mp4
                002.mp4
              swipe_down/
              swipe_left/
              swipe_right/
              click/

        Returns:
            A list of dictionaries where each dictionary represents one video sample.
        """

        samples: list[dict[str, str]] = []
        sample_index = 1

        for gesture_folder in sorted(self.samples_dir.iterdir()):
            if not gesture_folder.is_dir():
                continue

            label = gesture_folder.name
            video_files = self._find_video_files(gesture_folder)

            for video_path in video_files:
                sample_id = f"recorded_{sample_index:06d}"

                samples.append(
                    {
                        "sample_id": sample_id,
                        "source_type": "video",
                        "source_name": "recorded",
                        "external_id": "",
                        "label": self.label_mapper.converter_label(label),
                        "raw_label": label,
                        "path": self._format_path(video_path),
                    }
                )

                sample_index += 1

        return samples

    def _find_video_files(self, gesture_folder: Path) -> list[Path]:
        """Find all valid video files inside one gesture folder."""

        video_files = [
            file_path
            for file_path in gesture_folder.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in self.VALID_VIDEO_EXTENSIONS
        ]

        return sorted(video_files)

    def _format_path(self, path: Path) -> str:
        """
        Format path for samples.csv.

        If project_root is provided, save the path relative to the project root.
        Otherwise, save the path as given.
        """

        if self.project_root is None:
            return path.as_posix()

        return path.relative_to(self.project_root).as_posix()
