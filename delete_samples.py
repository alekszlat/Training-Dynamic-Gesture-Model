# Delete specified samples from the database
import csv
import os
from pathlib import Path

PATH_TO_SAMPLES_TO_DELETE = Path("data/manifests/test/user_0_manifest.csv")
PATH_TO_METADATA_TO_DELETE = Path("data/processed/test/user_0_metadata.csv")


def main():

    samples_to_delete = []
    landmark_to_delete = {}
    # Read the samples to delete from the manifest file
    if PATH_TO_SAMPLES_TO_DELETE.exists():
        with open(PATH_TO_SAMPLES_TO_DELETE, "r") as f:
            reader = csv.DictReader(f)
            samples_to_delete = [line["path"] for line in reader]
    else:
        print(f"Samples manifest file not found: {PATH_TO_SAMPLES_TO_DELETE}")

    if PATH_TO_METADATA_TO_DELETE.exists():
        with open(PATH_TO_METADATA_TO_DELETE, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                if line["path"] in samples_to_delete:
                    landmark_to_delete[line["path"]] = line["landmark_path"]
    else:
        print(f"Metadata file not found: {PATH_TO_METADATA_TO_DELETE}")

    for sample_path in samples_to_delete:
        print(f"Deleting sample: {sample_path}")

    for landmark_id, landmark_path in landmark_to_delete.items():
        print(f"Deleting landmark ID: {landmark_id}, Path: {landmark_path}")

    print(f"Samples manifest path: {PATH_TO_SAMPLES_TO_DELETE}")
    print(f"Number of samples to delete: {len(samples_to_delete)}")
    print(f"Number of landmarks to delete: {len(landmark_to_delete)}")

    for sample_path in samples_to_delete:
        if Path(sample_path).exists():
            os.remove(sample_path)
            print(f"Deleted sample: {sample_path}")
        else:
            print(f"Sample not found: {sample_path}")

    for landmark_id, landmark_path in landmark_to_delete.items():
        if Path(landmark_path).exists():
            os.remove(landmark_path)
            print(f"Deleted landmark: {landmark_path}")
        else:
            print(f"Landmark not found: {landmark_path}")

    os.remove(PATH_TO_SAMPLES_TO_DELETE)
    os.remove(PATH_TO_METADATA_TO_DELETE)


if __name__ == "__main__":
    main()
