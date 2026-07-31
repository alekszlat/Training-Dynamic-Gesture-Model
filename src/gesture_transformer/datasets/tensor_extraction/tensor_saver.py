from pathlib import Path
import json
import torch


class TensorSaver:
    """Saves processed train/validation tensors and label mappings."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_split(self, filename: str, data: dict) -> Path:
        """
        Save one dataset split.

        Args:
            filename: Example "train.pt" or "val.pt"
            data: Dictionary containing x, y, mask, sample_ids, labels

        Returns:
            Path to the saved file.
        """

        file_path = self.save_dir / filename

        torch.save(data, file_path)

        return file_path

    def save_label_mapping(
        self,
        mapping: dict[str, int],
        filename: str = "label_to_index.json",
    ) -> Path:
        """
        Save label-to-index mapping as JSON.

        Args:
            mapping: Example {"swipe_left": 0, "swipe_right": 1}
            filename: Output JSON filename

        Returns:
            Path to the saved file.
        """

        file_path = self.save_dir / filename

        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(
                mapping,
                fp,
                sort_keys=False,
                indent=4,
                ensure_ascii=False,
            )

        return file_path