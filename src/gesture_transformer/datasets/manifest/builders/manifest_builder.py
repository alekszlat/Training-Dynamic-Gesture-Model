from pathlib import Path
from gesture_transformer.datasets.manifest.label_mapper import LabelMapper
from dataclasses import dataclass
from  typing import Protocol


@dataclass(frozen=True)
class ManifestAttributes:
    sample_id:str
    source_type:str
    source_name:str
    external_id:str
    label:str
    raw_label:str
    path:str


class ManifestBuilder(Protocol):
    
    def __init__(
        self,
        data_location: str,
        samples_dir: Path,
        project_root: Path | None = None,
    ):
        self.samples_dir = samples_dir
        self.project_root = project_root

    def build(self) -> list[ManifestAttributes]:
        ...
