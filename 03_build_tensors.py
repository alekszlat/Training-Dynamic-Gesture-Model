from config import METADATA_PATH, PROCESSED_DIR, SUPPORTED_LABELS
from gesture_transformer.datasets.tensor_extraction.tensor_builder import (
    TensorBuilder,
)

LABEL_LIST = sorted(SUPPORTED_LABELS)

if __name__ == "__main__":
    builder = TensorBuilder(
        metadata_path=METADATA_PATH,
        processed_dir=PROCESSED_DIR,
        label_list=LABEL_LIST,
        target_length=40,
    )

    builder.build()
