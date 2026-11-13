from config import (
    METADATA_PATH,
    PROCESSED_DIR,
    SEEDS,
    SUPPORTED_LABELS,
    TARGET_SEQUENCE_LENGTH,
)
from gesture_transformer.datasets.tensor_extraction.tensor_builder import (
    TensorBuilder,
)

LABEL_LIST = sorted(SUPPORTED_LABELS)

if __name__ == "__main__":
    for seed in SEEDS:
        print(f"\n=== Building tensors for seed {seed} ===")
        builder = TensorBuilder(
            metadata_paths=METADATA_PATH,
            processed_dir=PROCESSED_DIR / f"seed_{seed}",
            label_list=LABEL_LIST,
            target_length=TARGET_SEQUENCE_LENGTH,
            random_seed=seed,
        )

        builder.build()
