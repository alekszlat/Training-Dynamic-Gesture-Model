import csv
from pathlib import Path


class ManifestCombiner:
    """Combine both the recorded and jester lists[dict] into single one and saves it into a csv file."""

    def __init__(
        self,
        recorded_list: list[dict],
        jester_list: list[dict],
        output_path: Path,
        supported_labels: set[str],
    ):
        self.recorded_list = recorded_list
        self.jester_list = jester_list
        self.output_path = output_path
        self.supported_labels = supported_labels

    def build_manifest(self) -> bool:
        combined_manifest = []
        combined_manifest.extend(self.recorded_list)
        combined_manifest.extend(self.jester_list)

        # Validate combined manifest rows
        validated_manifest = []
        invalid_rows = []
        for row in combined_manifest:
            check, message = self._validate_combined_manifest_row(row)

            if check:
                if row["label"] in self.supported_labels:
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

    def _validate_combined_manifest_row(self, row: dict) -> tuple[bool, str]:
        if "sample_id" not in row or not row["sample_id"]:
            return False, "missing sample_id"

        if "label" not in row or not row["label"]:
            return False, "missing label"

        if "path" not in row or not row["path"]:
            return False, "missing path"

        if not Path(row["path"]).exists():
            return False, "path does not exist"

        if "source_type" not in row:
            return False, "missing source_type"

        if row["source_type"] not in ["video", "jester"]:
            return False, "invalid source_type"

        return True, "valid row"

    def save_to_csv(self, manifest: list[dict[str, str]]) -> None:
        """Save the combined manifest to a CSV file."""

        if not manifest:
            print("Warning: No valid manifest rows to save.")
            return

        # Make sure the parent folder exists.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "source_type",
            "source_name",
            "external_id",
            "label",
            "raw_label",
            "path",
        ]

        with self.output_path.open(mode="w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

        print(f"Saved manifest to: {self.output_path}")

    def print_manifest_summary(self, manifest: list[dict[str, str]]) -> None:
        """Print a summary of the combined manifest."""

        total_samples = len(manifest)
        label_counts: dict[str, int] = {}

        for row in manifest:
            label = row.get("label")
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"Total samples in combined manifest: {total_samples}")
        print("Sample counts by label:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
