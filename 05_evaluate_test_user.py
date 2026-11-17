"""Evaluate the seeded models on the unseen test users (docs/evaluation-protocol.md)."""

import json
import statistics

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import (
    BEST_MODEL_FILENAME,
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    SEEDS,
    SUPPORTED_LABELS,
    TARGET_SEQUENCE_LENGTH,
)
from gesture_transformer.datasets.tensor_extraction.metadata_reader import (
    MetadataReader,
)
from gesture_transformer.datasets.tensor_extraction.tensor_builder import (
    TensorBuilder,
)
from gesture_transformer.training.gesture_transformer_en import GestureTransformer

TEST_DIR = PROCESSED_DIR / "test"
LABELS = sorted(SUPPORTED_LABELS)


def load_test_data() -> tuple[dict, list[str]]:
    """Build the test tensors the same way as training, and note the user of each clip."""

    builder = TensorBuilder(
        metadata_paths=[],
        processed_dir=TEST_DIR,
        label_list=LABELS,
        target_length=TARGET_SEQUENCE_LENGTH,
    )

    records, users = [], []
    for path in sorted(TEST_DIR.glob("user_*_metadata.csv")):
        user_records = MetadataReader([path])._load_metadata()
        records += user_records
        users += [path.stem.split("_")[1]] * len(user_records)

    return builder._build_split_tensors(records), users


def load_model(seed: int) -> GestureTransformer:
    """Load the best checkpoint of one seed, built with the settings it was trained with."""

    folder = CHECKPOINT_DIR / f"seed_{seed}"
    model_args = json.loads((folder / "model_args.json").read_text(encoding="utf-8"))

    model = GestureTransformer(**model_args)
    checkpoint = torch.load(
        folder / BEST_MODEL_FILENAME, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.eval()


@torch.no_grad()
def predict(model: GestureTransformer, data: dict) -> list[int]:
    # True marks a padded frame, the same convention as GestureDataset.
    padding_mask = data["mask"] == 0

    return model(data["x"], padding_mask).argmax(dim=1).tolist()


def main() -> None:
    data, users = load_test_data()
    y_true = data["y"].tolist()
    class_ids = list(range(len(LABELS)))

    print(f"Test clips: {len(y_true)}, users: {sorted(set(users))}")

    accuracies, macro_f1s = [], []

    for seed in SEEDS:
        y_pred = predict(load_model(seed), data)
        correct = [t == p for t, p in zip(y_true, y_pred)]

        accuracy = sum(correct) / len(correct)
        macro_f1 = f1_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        )
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)

        print(f"\n=== Seed {seed} ===")
        print(f"Accuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}")

        for user in sorted(set(users)):
            user_correct = [c for c, u in zip(correct, users) if u == user]
            print(f"User {user}: {sum(user_correct)}/{len(user_correct)} correct")

        print(
            classification_report(
                y_true,
                y_pred,
                labels=class_ids,
                target_names=LABELS,
                zero_division=0,
                digits=3,
            )
        )
        print(confusion_matrix(y_true, y_pred, labels=class_ids))

    print("\n=== All seeds (mean ± SD) ===")
    print(f"Accuracy: {statistics.mean(accuracies):.4f} ± {statistics.stdev(accuracies):.4f}")
    print(f"Macro-F1: {statistics.mean(macro_f1s):.4f} ± {statistics.stdev(macro_f1s):.4f}")


if __name__ == "__main__":
    main()