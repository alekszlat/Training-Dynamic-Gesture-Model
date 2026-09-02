from config import (
    METADATA_PATH,
    PROCESSED_DIR,
    SUPPORTED_LABELS,
    TARGET_SEQUENCE_LENGTH,
)
from gesture_transformer.datasets.tensor_extraction.tensor_builder import (
    TensorBuilder,
)

LABEL_LIST = sorted(SUPPORTED_LABELS)

if __name__ == "__main__":
    builder = TensorBuilder(
        metadata_path=METADATA_PATH,
        processed_dir=PROCESSED_DIR,
        label_list=LABEL_LIST,
        target_length=TARGET_SEQUENCE_LENGTH,
    )

    builder.build()

    # see how many samples gesture has
