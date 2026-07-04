from pathlib import Path
from typing import Iterator, Protocol, TypeAlias

import numpy as np

Frame: TypeAlias = np.ndarray

class FrameReader(Protocol):
    """A protocol for reading frames from a data source."""

    def read_frames(self, path: Path) -> Iterator[Frame]:
        """Read frames from the given path."""
        ...