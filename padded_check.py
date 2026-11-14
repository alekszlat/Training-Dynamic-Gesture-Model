import csv
import statistics
from collections import defaultdict

from config import METADATA_PATH, PROJECT_ROOT, TARGET_SEQUENCE_LENGTH
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataRecord,
)
from gesture_transformer.datasets.tensor_extraction.metadata_reader import (
    MetadataReader,
)

# Native frame rate of each source. Only used to turn frame counts into
# seconds: frame counts alone cannot be compared across sources.
SOURCE_FPS = {
    "jester": 12,
    "recorded": 30,
}

ALL_LABELS = "ALL"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "padding_report.csv"
COLUMNS = [
    "source",
    "label",
    "n",
    "padded_pct",
    "squeezed_pct",
    "median_frames",
    "min_frames",
    "max_frames",
    "median_seconds",
]


def group_frame_counts(
    records: list[MetadataRecord],
) -> dict[tuple[str, str], list[int]]:
    """
    Collect total_frames per (source, label), plus one ALL group per source.

    Args:
        records: Usable metadata records.

    Returns:
        Frame counts keyed by (source_name, label).
    """

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)

    for record in records:
        groups[(record.source_name, record.label)].append(record.total_frames)
        groups[(record.source_name, ALL_LABELS)].append(record.total_frames)

    return groups


def summarize(source: str, label: str, frame_counts: list[int]) -> dict:
    """
    Compute the padding and length statistics for one group.

    Args:
        source: Source name, a key of SOURCE_FPS.
        label: Gesture label, or ALL_LABELS.
        frame_counts: total_frames of every clip in the group.

    Returns:
        One report row keyed by COLUMNS.
    """

    n = len(frame_counts)
    padded = sum(count < TARGET_SEQUENCE_LENGTH for count in frame_counts)
    squeezed = sum(count > TARGET_SEQUENCE_LENGTH for count in frame_counts)
    median = float(statistics.median(frame_counts))

    return {
        "source": source,
        "label": label,
        "n": n,
        "padded_pct": round(100 * padded / n, 1),
        "squeezed_pct": round(100 * squeezed / n, 1),
        "median_frames": median,
        "min_frames": min(frame_counts),
        "max_frames": max(frame_counts),
        "median_seconds": round(median / SOURCE_FPS[source], 2),
    }


def sort_key(row: dict) -> tuple:
    """Sort by source, then label, with the ALL row last in each source."""

    return (row["source"], row["label"] == ALL_LABELS, row["label"])


def print_table(rows: list[dict]) -> None:
    """Print the report rows as an aligned table."""

    header = (
        f"{'source':<10}{'label':<20}{'n':>6}{'padded %':>10}{'squeezed %':>12}"
        f"{'median':>8}{'min':>6}{'max':>6}{'median s':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['source']:<10}{row['label']:<20}{row['n']:>6}"
            f"{row['padded_pct']:>10}{row['squeezed_pct']:>12}"
            f"{row['median_frames']:>8}{row['min_frames']:>6}{row['max_frames']:>6}"
            f"{row['median_seconds']:>10}"
        )


def write_csv(rows: list[dict]) -> None:
    """Save the report rows to OUTPUT_PATH."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = MetadataReader(METADATA_PATH)._load_metadata()
    groups = group_frame_counts(records)

    rows = sorted(
        (
            summarize(source, label, counts)
            for (source, label), counts in groups.items()
        ),
        key=sort_key,
    )

    print_table(rows)
    write_csv(rows)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
