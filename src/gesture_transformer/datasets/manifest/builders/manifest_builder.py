from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ManifestAttributes:
    sample_id: str
    source_type: str
    source_name: str
    external_id: str
    label: str
    raw_label: str
    path: str


class ManifestBuilder(Protocol):
    def build(self) -> list[ManifestAttributes]: ...
