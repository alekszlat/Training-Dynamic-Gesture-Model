"""
Checks that every landmark file recorded_metadata.csv references is in git.

A row and its .npy are written to different folders, so it is easy to commit one
without the other. That looks fine to the author, whose files are on disk, and
breaks for whoever pulls.

Author:
    Hristo Hristov
"""

import csv
import subprocess
import sys
from pathlib import Path

METADATA_PATH = Path("data/processed/recorded_metadata.csv")
LANDMARKS_DIR = "data/interim/landmarks/recorded"


def untracked_landmarks(metadata_path: Path) -> list[str]:
    """
    Find landmark files the metadata references but git does not have.

    Args:
        metadata_path: Metadata csv to check.

    Returns:
        Repository-relative paths, sorted, empty when nothing is missing.
    """

    tracked = subprocess.run(
        ["git", "ls-files", "--cached", LANDMARKS_DIR],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()

    with metadata_path.open(newline="", encoding="utf-8") as file:
        referenced = {
            row["landmark_path"] for row in csv.DictReader(file) if row["landmark_path"]
        }

    return sorted(referenced - set(tracked))


def main() -> int:
    """
    Report any landmark file the metadata references but git does not have.

    Returns:
        0 when nothing is missing, 1 otherwise.
    """

    if not METADATA_PATH.exists():
        return 0

    missing = untracked_landmarks(METADATA_PATH)

    if not missing:
        return 0

    print(f"{METADATA_PATH} references {len(missing)} landmark file(s) not in git:\n")
    for path in missing[:5]:
        print(f"    git add {path}")
    if len(missing) > 5:
        print(f"    ... and {len(missing) - 5} more")

    return 1


if __name__ == "__main__":
    sys.exit(main())
