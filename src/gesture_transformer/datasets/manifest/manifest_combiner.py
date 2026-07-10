from pathlib import Path
import csv

SUPPORTED_LABELS = {
    "swiping_left",
    "swiping_right",
    "swiping_up",
    "swiping_down",
    "doing_other_things",
    "click",
}

class ManifestCombiner:
    """Combine both the recorded and jester lists[dict] into single one and saves it into a csv file."""

    def __init__(self, recorded_list: list[dict], jester_list: list[dict], output_path: Path):
        self.recorded_list = recorded_list
        self.jester_list = jester_list
        self.output_path = output_path

    def build_manifest(self) -> bool:
        combined_manifest = []
        combined_manifest.extend(self.recorded_list)
        combined_manifest.extend(self.jester_list)

        # Validate combined manifest rows
        validated_manifest = []
        for row in combined_manifest:
            if self._validate_combined_manifest_row(row):
                validated_manifest.append(row)
            else:
                print(f"Warning: Invalid manifest row skipped: {row}")
                return False # Return False if any invalid row is found
        
        self.save_to_csv(validated_manifest)
        self.print_manifest_summary(validated_manifest)
        
        return True

    def _validate_combined_manifest_row(self, row: dict) -> bool:
        if "sample_id" not in row or not row["sample_id"]:
            print("Invalid row: missing sample_id")
            return False

        if "label" not in row or not row["label"]:
            print("Invalid row: missing label")
            return False

        if row["label"] not in SUPPORTED_LABELS:
            print(f"Invalid row: unsupported label {row['label']}")
            return False

        if "path" not in row or not row["path"]:
            print("Invalid row: missing path")
            return False

        if not Path(row["path"]).exists():
            print(f"Invalid row: path does not exist: {row['path']}")
            return False

        if "source_type" not in row:
            print("Invalid row: missing source_type")
            return False

        if row["source_type"] not in ["video", "jester"]:
            print(f"Invalid row: invalid source_type {row['source_type']}")
            return False

        return True


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
        label_counts = {}

        for row in manifest:
            label = row.get("label")
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"Total samples in combined manifest: {total_samples}")
        print("Sample counts by label:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")