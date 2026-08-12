from pathlib import Path

from src.gesture_transformer.datasets.tensor_extraction.tensor_builder import (
    TensorBuilder,
)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent

    METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    LABEL_LIST = [
        "swiping_down",
        "swiping_left",
        "swiping_right",
        "swiping_up",
        "click",
    ]

    builder = TensorBuilder(
        metadata_path=METADATA_PATH,
        processed_dir=PROCESSED_DIR,
        label_list=LABEL_LIST,
        target_length=40,
    )

    builder.build()
