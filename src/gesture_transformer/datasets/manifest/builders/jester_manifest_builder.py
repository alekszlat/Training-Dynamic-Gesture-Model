import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from gesture_transformer.datasets.manifest.builders.manifest_builder import (
    ManifestAttributes,
)
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper


@dataclass(frozen=True)
class JesterAnnotation:
    """Represents one raw annotation row from the Jester dataset."""

    external_id: str
    raw_label: str


class JesterManifestBuilder:
    """Build manifest rows for Jester gesture frame folders."""

    VALID_FRAME_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png"}

    ANNOTATION_FILE = "labeled_data.csv"

    def __init__(
        self,
        data_location: str,
        samples_dir: Path,
        project_root: Path | None = None,
    ):
        self.data_location = data_location
        self.samples_dir = samples_dir
        self.project_root = project_root
        self.label_mapper = LabelMapper()

    def build(self) -> list[ManifestAttributes]:
        """
        Build manifest rows for supported Jester gesture samples.

        Expected Jester structure:

            20bn-jester-v1/
              annotations/
                DATA_LABELED.csv
              frames/
                1/
                  00001.jpg
                  00002.jpg
                2/
                ...
        """

        annotations = self._read_all_annotations()

        samples: list[ManifestAttributes] = []
        sample_index = 1

        for annotation in annotations:
            internal_label = self.label_mapper.converter_label(annotation.raw_label)
            if internal_label is None:
                continue

            frame_folder = self._create_frame_folder_path(annotation.external_id)
            if self.data_location == "local" and not self._has_frames(frame_folder):
                continue

            if self.data_location == "local":
                sample_id = f"in_jester_{sample_index:06d}"
            else:
                sample_id = f"ex_jester_{sample_index:06d}"

            samples.append(
                ManifestAttributes(
                    sample_id=sample_id,
                    source_type="jester",
                    source_name="jester",
                    external_id=annotation.external_id,
                    label=internal_label,
                    raw_label=annotation.raw_label,
                    path=self._format_path(frame_folder),
                )
            )
            sample_index += 1

        return samples

    def _read_all_annotations(self) -> list[JesterAnnotation]:
        """Read all labeled Jester annotation files used by this project."""

        annotations: list[JesterAnnotation] = []

        annotation_file = self.samples_dir / "annotations" / self.ANNOTATION_FILE
        annotations.extend(self._read_annotation_file(annotation_file))

        return annotations

    def _read_annotation_file(self, annotation_file: Path) -> list[JesterAnnotation]:
        """
        Read one Jester annotation CSV file.

        Expected row format:
            video_id;label

        Example:
            12345;Swiping Left
        """

        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        annotations: list[JesterAnnotation] = []

        with annotation_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(";")

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid Jester annotation row in {annotation_file}: {line}"
                    )

                external_id, raw_label = parts

                annotations.append(
                    JesterAnnotation(
                        external_id=external_id,
                        raw_label=raw_label,
                    )
                )

        return annotations

    def _create_frame_folder_path(self, external_id: str) -> Path:
        """Create the path to the Jester frame folder for one video ID."""

        return self.samples_dir / "frames" / external_id

    def _has_frames(
        self,
        frame_folder: Path,
    ) -> bool:
        try:
            with os.scandir(frame_folder) as entries:
                return any(
                    entry.is_file()
                    and Path(entry.name).suffix.lower() in self.VALID_FRAME_EXTENSIONS
                    for entry in entries
                )

        except (FileNotFoundError, NotADirectoryError):
            return False

    def _format_path(self, path: Path) -> str:
        """
        Format the path for samples.csv.

        Prefer paths relative to the project root when project_root is provided.
        """

        if self.project_root is None:
            return path.as_posix()

        return path.relative_to(self.project_root).as_posix()

    def _to_jester_label(self, label: str) -> str:
        return label.replace("_", " ").title()
