from csv import DictWriter
from dataclasses import asdict, fields
from pathlib import Path

from gesture_transformer.datasets.manifest.builders.manifest_builder import (
    ManifestAttributes,
)


class ManifestCombiner:
    """Combine both the recorded and jester lists[dict] into single one and saves it into a csv file."""

    def __init__(
        self,
        datasets_list: list[list[ManifestAttributes]],
        output_path: Path,
        supported_labels: set[str],
    ):
        self.datasets_list = datasets_list
        self.output_path = output_path
        self.supported_labels = supported_labels

    def build_manifest(self) -> bool:
        combined_manifest: list[ManifestAttributes] = []
        counter = 1

        for dataset in self.datasets_list:
            if not dataset:
                print(f"Dataset_{counter} is empty")
                counter += 1
                continue

            combined_manifest.extend(dataset)
            counter += 1

        # Validate combined manifest rows
        validated_manifest = []
        invalid_rows = []

        print("Going over combined list")
        for row in combined_manifest:
            check, message = self._validate_combined_manifest_row(row)

            if check:
                ##print(message)
                if row.label in self.supported_labels:
                    validated_manifest.append(row)
                else:
                    continue
            else:
                invalid_rows.append((row, message))

        if invalid_rows:
            print(
                f"Warning: {len(invalid_rows)} invalid rows found in the combined manifest."
            )
            for invalid_row, message in invalid_rows:
                print(f"Invalid row: {invalid_row} - {message}")
            return False

        self.save_to_csv(validated_manifest)
        self.print_manifest_summary(validated_manifest)

        return True

    def _validate_combined_manifest_row(
        self,
        row: ManifestAttributes,
    ) -> tuple[bool, str]:
        if not row.sample_id:
            return False, "missing sample_id"

        if not row.label:
            return False, "missing label"

        if not row.path:
            return False, "missing path"

        if not Path(row.path).exists():
            return False, "path does not exist"

        if not row.source_type:
            return False, "missing source_type"

        if row.source_type not in {"video", "jester"}:
            return False, "invalid source_type"

        return True, "valid row"

    def save_to_csv(
        self,
        manifest: list[ManifestAttributes],
    ) -> None:
        """Save the combined manifest to a CSV file."""

        if not manifest:
            print("Warning: No valid manifest rows to save.")
            return

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [field.name for field in fields(ManifestAttributes)]

        with self.output_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            writer = DictWriter(
                csvfile,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(asdict(row) for row in manifest)

        print(f"Saved manifest to: {self.output_path}")

    def print_manifest_summary(
        self,
        manifest: list[ManifestAttributes],
    ) -> None:
        """Print a summary of the combined manifest."""

        total_samples = len(manifest)
        label_counts: dict[str, int] = {}

        for row in manifest:
            label = row.label

            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"Total samples in combined manifest: {total_samples}")
        print("Sample counts by label:")

        for label, count in label_counts.items():
            print(f"  {label}: {count}")
